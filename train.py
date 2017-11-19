"""train.py
Usage:
    train.py <f_model_config> <f_opt_config>  [--prefix <p>] [--ce] [--db]
    train.py -r <exp_name> <idx> [--test]
Arguments:
 
Example:
    right now there are 2 models
    'fc' for fully-connect NNs
    'lr' for Logistic regression 
    and 3 optimizer
    'sgd' ...
    'nsgd' sgd with added Gaussian Noise, but learn-rate not decayed properly
    'sgld' sgd with added Gaussian Noise, and both decayed polynomially
    python train.py model/config/lr.yaml opt/config/sgd-128-lr.yaml --ce
    python train.py model/config/lr.yaml opt/config/nsgd-128-lr.yaml --ce
    python train.py model/config/lr.yaml opt/config/sgld-128-lr.yaml --ce
Options:
"""
from __future__ import division
"""
known todos:
1. check the threshold for starting to collect samples
2. don't do "model.model.state_dict", write a wrapper for each model class
3. right now only keeping <sample_size> posterior samples to prevent memory problems... fix this
"""
import copy
from docopt import docopt
import yaml
import torch
import tensorflow as tf
from torch import optim
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import os
import numpy as np

from utils import MiniBatcher#, MiniBatcherPerClass
import torchvision.models as tvm
import datetime
from opt.loss import *
from opt.nsgd import NoisedSGD
from model.fc import fc, lr
from model.cnn import *
# from model.cnn import *
import cPickle as pkl
from cleverhans.utils_mnist import data_mnist

from tensorboard_monitor.configuration import*
from tensorboard_monitor.monitor import*

#### magic
NUM_VALID =10000 
tft = lambda x:torch.FloatTensor(x)
tfv = lambda x:Variable(tft(x))

def main(arguments):
    if arguments['-r']:
        exp_name = arguments['<exp_name>']
        f_model_config = 'model/config/'+exp_name[exp_name.find(':')+1:].split('-X-')[0]+'.yaml'
        f_opt_config = 'opt/config/'+exp_name[exp_name.find(':')+1:].split('-X-')[1]+'.yaml'
        old_exp_name = exp_name
        exp_name += '_resumed'
    else:
        f_model_config = arguments['<f_model_config>']
        f_opt_config = arguments['<f_opt_config>']
        model_name = os.path.basename(f_model_config).split('.')[0]
        opt_name = os.path.basename(f_opt_config).split('.')[0]
        timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
        data_name = 'mnist'
        if arguments['--prefix']:
            exp_name = '%s:%s-X-%s-X-%s@%s' % (arguments['<p>'], model_name, opt_name, data_name, timestamp)
        else:
            exp_name = '%s-X-%s-X-%s@%s' % (model_name, opt_name, data_name, timestamp)
        if arguments['--ce']:
            exp_name = 'CE.' + exp_name
        
    
    model_config = yaml.load(open(f_model_config, 'rb'))
    opt_config = yaml.load(open(f_opt_config, 'rb'))

    print ('\n\n\n\n>>>>>>>>> [Experiment Name]')
    print (exp_name)
    print ('<<<<<<<<<\n\n\n\n')

    ## Experiment stuff
    if not os.path.exists('./saves/%s'%exp_name):
        os.makedirs('./saves/%s'%exp_name)

    ## Data
    X, Y, X_test, Y_test = data_mnist() #(N, W, H, C) ...  
    Y_val = Y[-NUM_VALID:]
    X_val = X[-NUM_VALID:]
    Y = Y[:np.min([opt_config['dataset_size'],NUM_VALID])]
    X = X[:np.min([opt_config['dataset_size'],NUM_VALID])]
    
    X = tfv(X)
    Y = tft(Y)
    # Dataset (X size(N,D) , Y size(N,K))
    ## Model
    model = eval(model_config['name'])(**model_config['kwargs'])
    model.type(torch.FloatTensor)
    ## Optimizer
    opt = eval(opt_config['name'])(model.parameters(), **opt_config['kwargs'])


    if arguments['-r']:
        model.load('./saves/%s/model_%s.t7'%(old_exp_name,arguments['<idx>']))
        opt.load_state_dict(torch.load('./saves/%s/opt_%s.t7'%(old_exp_name,arguments['<idx>'])))

        if arguments['--test']:
            raise NotImplementedError()


    cross_e = Monitor('cross_e')
    accuracy = Monitor('accuracy')
    sess, train_writer, point_writer, posterior_writer = initial_tensorboard(exp_name)

    # ## tensorboard
    # #ph
    # ph_accuracy = tf.placeholder(tf.float32,  name='accuracy')
    # ph_loss = tf.placeholder(tf.float32,  name='loss')
    # if not os.path.exists('./logs'):
    #     os.mkdir('./logs')
    # tf_acc = tf.summary.scalar('accuracy', ph_accuracy)
    # tf_loss = tf.summary.scalar('loss', ph_loss)
    #
    # log_folder = os.path.join('./logs', exp_name)
    # # remove existing log folder for the same model.
    # if os.path.exists(log_folder):
    #     import shutil
    #     shutil.rmtree(log_folder, ignore_errors=True)
    #
    # sess = tf.InteractiveSession()
    #
    # train_writer = tf.summary.FileWriter(os.path.join(log_folder, 'train'), sess.graph)
    # val_writer = tf.summary.FileWriter(os.path.join(log_folder, 'val'), sess.graph)
    #
    batcher = eval(opt_config['batcher_name'])(X.size()[0], **opt_config['batcher_kwargs'])

    ## Loss
    if arguments['--ce']:
        Loss = CE()
    else:
        raise NotImplementedError()
    

    best_val_acc = 0
    val_errors = []
    tf.global_variables_initializer().run()
    posterior_samples =[]
    posterior_weights = []
    is_collecting = False
    alpha_thresh = 0.01
    sample_size = 500
    sample_interval = 100
    if not arguments['--db']:
        ## Algorithm
        sd = opt.state_dict()
        step_size = sd['param_groups'][0]['lr']     
        for idx in tqdm(xrange(opt_config['max_train_iters'])):
        # for idx in (xrange(opt_config['max_train_iters'])):
            if 'lrsche' in opt_config and opt_config['lrsche'] != [] and opt_config['lrsche'][0][0] == idx:
                _, tmp_fac = opt_config['lrsche'].pop(0)
                sd = opt.state_dict()
                assert len(sd['param_groups']) ==1
                sd['param_groups'][0]['lr'] *= tmp_fac
                opt.load_state_dict(sd)
            if idx > 0 and  'lrpoly' in opt_config :
                a, b, g = opt_config['lrpoly']
                sd = opt.state_dict()
                step_size = a*((b+idx)**(-g))
                sd['param_groups'][0]['lr'] = step_size
                opt.load_state_dict(sd)

            idxs = batcher.next(idx)
            X_batch = X[torch.LongTensor(idxs)].type(torch.FloatTensor)
            Y_batch = Y[torch.LongTensor(idxs)]#.type(torch.cuda.FloatTensor)
            ## network
            tv_F = model.forward(X_batch)
            F = tv_F.data.clone().type(torch.FloatTensor)
            ### loss layer
            loss, G, train_pred = Loss.train(F, Y_batch)

            model.zero_grad()
            tv_F.backward(gradient=G.type(torch.FloatTensor),retain_variables=True)
            opt.step()


            # TensorBoard
            #accuracy
            train_gt = Y[torch.LongTensor(idxs)].numpy().argmax(1)
            train_accuracy = (train_pred[batcher.start_unlabelled:] == train_gt[batcher.start_unlabelled:]).mean()
            accuracy.record_tensorboard(train_accuracy, idx, sess, train_writer)
            cross_e.record_tensorboard(loss, idx, sess, train_writer)
            # # summarize
            # acc= sess.run(tf_acc, feed_dict={ph_accuracy:train_accuracy})
            # loss = sess.run(tf_loss, feed_dict={ph_loss:loss})
            # tmp = Y_batch.numpy()
            # train_writer.add_summary(acc+loss, idx)
            #
            # ce_bayes = Monitor('bayesian')
            # sess, train_writer, point_writer, posterior_writer = initial_tensorboard(exp_name)

            ## monitor gradient variance
            ## TODO: this is extremely stupid...
            if 'NoisedSGD' in str(opt.__class__) and is_collecting ==False and  idx>0 and idx%sample_interval==0:
                # oG = G.type(torch.cuda.FloatTensor)
                # gs = []
                # for gidx in xrange(oG.size()[0]):
                #     model.zero_grad()
                #     tv_F.backward(gradient=oG[gidx][None].repeat(batcher.batch_size,1),retain_variables=True)
                #     ps = list(model.parameters())
                #     g = torch.cat([p.grad.view(-1,1) for p in ps])
                #     gs.append(g)
                # gs = torch.cat(gs,1).t() ## (N, D) gradient matrix
                # gs_centered = gs - gs.mean(0).expand_as(gs)
                # V_s = (1/opt.correction)*torch.mm(gs_centered,gs_centered.t()) ## (D, D), hopefully V_s in the paper
                # _,s,_ = torch.svd(V_s.data)
                # alpha = s[0] * step_size * opt.correction / batcher.batch_size
                # # print ("variance")
                # # print(s[0])
                # # print ("alpha")
                # # print (alpha)
                # # if alpha < alpha_thresh and idx > 5000: #todo: ... 
                if idx > 1000: #todo: ... 
                    is_collecting = True

            if is_collecting  and idx%sample_interval==0: 
                posterior_samples.append(copy.deepcopy(model.state_dict()))
                posterior_weights.append(step_size)
                if len(posterior_samples) > sample_size:
                    del posterior_samples[0]
                    del posterior_weights[0]
            #validate
            if idx>0 and idx%500==0:
                curr_state = model.state_dict()




                def _validate_batch(model, X_val_batch, Y_val_batch):
                    model.eval()
                    val_pred = Loss.infer(model, Variable(torch.FloatTensor(X_val_batch)).type(torch.FloatTensor))
                    val_accuracy = np.mean(Y_val_batch.argmax(1) == val_pred)
                    model.train()
                    return val_accuracy
                


                val_batch_size = batcher.batch_size
                val_batches = Y_val.shape[0] // val_batch_size
                v1 = []



                for vidx in xrange(val_batches):
                    val_accuracy = _validate_batch(model, X_val[vidx*val_batch_size:(vidx+1)*val_batch_size], Y_val[vidx*val_batch_size:(vidx+1)*val_batch_size])
                    v1.append(val_accuracy)


                def _validate_batch_bayes(posterior_samples,posterior_weights, X_val_batch, Y_val_batch):
                    model.eval()
                    acc_proba = None
                    # loss_sum = 0
                    for sample_idx in xrange(len(posterior_samples)):
                        p_sample = posterior_samples[sample_idx]
                        model.load_state_dict(p_sample)
                        _,proba = Loss.infer(model, Variable(torch.FloatTensor(X_val_batch)).type(torch.FloatTensor),  ret_proba=True)


                        if acc_proba is None:
                            acc_proba = posterior_weights[sample_idx] * proba
                        else:
                            acc_proba += posterior_weights[sample_idx] * proba
                    model.train()
                    val_pred = acc_proba.argmax(1)    
                    val_accuracy = np.mean(Y_val_batch.argmax(1) == val_pred)

                    return val_accuracy





                bayes_v = []
                if is_collecting:
                    for vidx in xrange(val_batches):
                        val_accuracy = _validate_batch_bayes(posterior_samples[-sample_size:],posterior_weights[-sample_size:], X_val[vidx*val_batch_size:(vidx+1)*val_batch_size], Y_val[vidx*val_batch_size:(vidx+1)*val_batch_size])
                        bayes_v.append(val_accuracy)



                val_accuracy = np.mean(v1)
                bayes_acc = np.mean(bayes_v)
                accuracy.record_tensorboard(val_accuracy, idx, sess, point_writer)
                accuracy.record_tensorboard(bayes_acc, idx, sess, posterior_writer)




                # print (val_accuracy, bayes_acc)
                # acc= sess.run(tf_acc, feed_dict={ph_accuracy:val_accuracy})
                # val_writer.add_summary(acc, idx)
                # val_errors.append(val_accuracy)
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    name = './saves/%s/model_best.t7'%(exp_name)
                    print ("[Saving to]")
                    print (name)
                    model.save(name)
                model.load_state_dict(curr_state)
            ## checkpoint
            if idx>0 and idx%1000==0:
                name = './saves/%s/model_%i.t7'%(exp_name,idx)
                print ("[Saving to]")
                print (name)
                model.save(name)
                torch.save(opt.state_dict(), './saves/%s/opt_%i.t7'%(exp_name,idx))
    #pkl.dump(val_errors, open(os.path.join(log_folder, 'val.log'), 'wb'))
    return best_val_acc


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    main(arguments)