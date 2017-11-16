"""train_new.py
Usage:
    train_new.py <f_model_config> <f_opt_config> <dataset> [--prefix <p>] [--ce] [--db] [--cuda]
    train_new.py -r <exp_name> <idx> [--test]

Arguments:

Example:
    right now there are 2 models
    'fc' for fully-connect NNs
    'lr' for Logistic regression
    and 3 optimizer
    'sgd' ...
    'nsgd' sgd with added Gaussian Noise, but learn-rate not decayed properly
    'sgld' sgd with added Gaussian Noise, and both decayed polynomially

    python train_new.py model/config/fc1-100.yaml opt/config/nsgd-bdk.yaml babymnist --ce --cuda

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

import datetime
from opt.loss import *
from model.fc import *
from opt.nsgd import NoisedSGD
import torch.optim as optim
from dataset.BabyMnist import BabyMnist

# from model.cnn import *
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from tensorboard_monitor.configuration import*
from tensorboard_monitor.monitor import*
from tqdm import tqdm

def load_configuration(arguments, name_dataset):
    if arguments['-r']:
        exp_name = arguments['<exp_name>']
        f_model_config = 'model/config/' + exp_name[exp_name.find(':') + 1:].split('-X-')[0] + '.yaml'
        f_opt_config = 'opt/config/' + exp_name[exp_name.find(':') + 1:].split('-X-')[1] + '.yaml'
        old_exp_name = exp_name
        exp_name += '_resumed'
    else:
        f_model_config = arguments['<f_model_config>']
        f_opt_config = arguments['<f_opt_config>']
        model_name = os.path.basename(f_model_config).split('.')[0]
        opt_name = os.path.basename(f_opt_config).split('.')[0]
        timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
        data_name = name_dataset
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
    if not os.path.exists('./saves/%s' % exp_name):
        os.makedirs('./saves/%s' % exp_name)

    ## Model
    model = eval(model_config['name'])(**model_config['kwargs'])
    if arguments['--cuda']:
        model.type(torch.cuda.FloatTensor)
    ## Optimizer
    opt = eval(opt_config['name'])(model.parameters(), **opt_config['kwargs'])
    ## loss
    if arguments['--ce']:
        Loss = CE()
    else:
        raise NotImplementedError()

    if arguments['-r']:
        model.load('./saves/%s/model_%s.t7' % (old_exp_name, arguments['<idx>']))
        opt.load_state_dict(torch.load('./saves/%s/opt_%s.t7' % (old_exp_name, arguments['<idx>'])))
        if arguments['--test']:
            raise NotImplementedError()

    return model, opt, Loss, exp_name, model_config, opt_config

def update_LearningRate(optimizer, iteration, opt_config):
    sd = optimizer.state_dict()
    learning_rate = sd['param_groups'][0]['lr']
    if 'lrsche' in opt_config and opt_config['lrsche'] != [] and opt_config['lrsche'][0][0] == iteration:
        _, tmp_fac = opt_config['lrsche'].pop(0)
        sd = optimizer.state_dict()
        assert len(sd['param_groups']) == 1
        sd['param_groups'][0]['lr'] *= tmp_fac
        optimizer.load_state_dict(sd)
    if iteration > 0 and 'lrpoly' in opt_config:
        a, b, g = opt_config['lrpoly']
        sd = optimizer.state_dict()
        step_size = a * ((b + iteration) ** (-g))
        sd['param_groups'][0]['lr'] = step_size
        optimizer.load_state_dict(sd)
    return learning_rate

def check_point(model, optimizer, iteration, exp_name):
    if iteration > 0 and iteration % 1000 == 0:
        name = './saves/%s/model_%i.t7' % (exp_name, iteration)
        print ("[Saving to]")
        print (name)
        model.save(name)
        torch.save(optimizer.state_dict(), './saves/%s/opt_%i.t7' % (exp_name, iteration))

def inference_accuracy(prediction, labels):
    accuracy = (prediction == labels.data.numpy()).mean().astype(float)
    return accuracy

def posterior_expectation(model,Loss, posterior_samples, posterior_weights, inputs ):
    num_posterior_samples = len(posterior_samples)
    outputs_weighted_sum = 0
    model.eval()
    for sample_idx in range(num_posterior_samples):
        model.model.load_state_dict(posterior_samples[sample_idx])
        outputs_samples =  model.forward(inputs)
        outputs_prob = Loss.softmax_output(outputs_samples)
        # outputs_sample --> softmax --> prob
        outputs_weighted_sum = outputs_weighted_sum + outputs_prob*posterior_weights[sample_idx]
    model.train()
    outputs_expectation = outputs_weighted_sum / (sum(posterior_weights))
    return outputs_expectation #(n,d)

def is_into_langevin_dynamics(model,outputs, gradient_data, optimizer, learning_rate,alpha_threshold,iteration,sess, train_writer):
    is_collecting = False
    gs = []
    oG = gradient_data.type(torch.FloatTensor)
    batch_size = oG.size()[0]
    num_parameter = oG.size()[1]
    for gidx in xrange(oG.size()[0]):
        model.zero_grad()
        # each example's gradient
        gradient_feed = batch_size * torch.cat((oG[gidx][None],torch.zeros(batch_size-1, num_parameter)))
        outputs.backward(gradient=gradient_feed, retain_variables=True)
        ps = list(model.parameters())
        g = torch.cat([p.grad.view(-1, 1) for p in ps])
        gs.append(g)

    gs = torch.cat(gs, 1).t()  ## (N, D) gradient matrix
    gs_centered = gs - gs.mean(0).expand_as(gs)
    V_s = (1 / optimizer.correction) * torch.mm(gs_centered, gs_centered.t())  ## (D, D), hopefully V_s in the paper
    _, s, _ = torch.svd(V_s.data)
    alpha = s[0] * (learning_rate/optimizer.correction) * (optimizer.correction* optimizer.correction) / (4*batch_size)

    if alpha < alpha_threshold:  # todo: ...
        is_collecting = True
    variance_monitor.record_tensorboard(np.log10(s[0]), iteration,sess, train_writer)
    return is_collecting

def posterior_sampling(sample_size, model,learning_rate, posterior_samples, posterior_weights):
    posterior_samples.append(copy.deepcopy(model.model.state_dict()))
    posterior_weights.append(learning_rate)
    if len(posterior_samples) > sample_size:
        del posterior_samples[0]
        del posterior_weights[0]
    return posterior_samples, posterior_weights

def main(arguments):

    # Set up
    iteration = 0
    batch_size = 100
    # alpha_threshold = .5
    burnin_iters= 5000
    sample_size = 100
    sample_interval = 20
    posterior_samples = []
    posterior_weights = []
    validation_interval = 2000
    variance_monitor_interval = 50
    name_dataset = arguments['<dataset>']
    is_collecting=False

  # Load argumentts: Module, Optimizer, Loss_function
    model, optimizer, Loss, exp_name, model_config, opt_config = load_configuration(arguments, name_dataset)
    num_max_iteration = opt_config['max_train_iters']
    log_folder = os.path.join('./logs/new', exp_name)

  # Load DataSet
    # trainLoader automatically generate training_batch
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if name_dataset == 'babymnist':
        trainset = BabyMnist( train=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testset = BabyMnist( train=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    if name_dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

   # Tensorboard
    # Set tensorboard_monitor monitors
    loss_monitor = Monitor('cross_entropy')
    accuracy_monitor = Monitor('accuracy')
    learning_rate_monitor = Monitor('learning_rate')
    variance_monitor = Monitor('variance')
    # Initial tensorboard
    sess, train_writer, point_writer, posterior_writer = initial_tensorboard(exp_name)

   # Training
    while(1):
        #print(len(posterior_samples))
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            if arguments['--cuda']:
                inputs = inputs.type(torch.cuda.FloatTensor)
            inputs, labels = Variable(inputs), Variable(labels)

            # Update learning rate
            learning_rate = update_LearningRate(optimizer, iteration, opt_config)
            learning_rate_monitor.record_tensorboard(learning_rate,iteration,sess,train_writer)

            # Inference
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            gradient_data = Loss.store_gradient_data(outputs,labels)

            # Loss function
            training_loss = Loss.nll_loss(outputs,labels)
            loss_monitor.record_tensorboard(training_loss.data[0],iteration,sess,train_writer)


            training_predictions = Loss.inference_prediction(outputs)
            accuracy = inference_accuracy(training_predictions, labels)
            accuracy_monitor.record_tensorboard(accuracy, iteration,sess, train_writer)
            accuracy_monitor.record_matplot( 1 - accuracy, iteration, 'train')

            # Parameter Update
            optimizer.zero_grad()
            if arguments['--cuda']:
                gradient_data = gradient_data.type(torch.cuda.FloatTensor)
            outputs.backward(gradient = gradient_data, retain_variables = True)
            optimizer.step()

            # monitor Variance
            if opt_config['name'] == 'NoisedSGD' and is_collecting==False and iteration % variance_monitor_interval == 0 :
                # is_collecting = is_into_langevin_dynamics(model, outputs,  gradient_data, optimizer, learning_rate,alpha_threshold,iteration,sess, train_writer)
                is_collecting = iteration >= burnin_iters
            if is_collecting == True and iteration %sample_interval == 0:
                posterior_samples, posterior_weights = posterior_sampling(sample_size, model, learning_rate, posterior_samples,
                                                                          posterior_weights)
           # Validation
            # Point Estimation
            if iteration % validation_interval == 0:
                # Load data
                dataiter = iter(testloader)
                test_inputs, test_labels = dataiter.next()
                if arguments['--cuda']:
                    test_inputs = test_inputs.type(torch.cuda.FloatTensor)
                test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
                # Inference
                point_outputs = model.forward(test_inputs)
                point_loss = Loss.nll_loss(point_outputs, test_labels)
                loss_monitor.record_tensorboard(point_loss.data[0], iteration, sess, point_writer)
                point_predictions = Loss.inference_prediction(point_outputs)
                point_accuracy = inference_accuracy(point_predictions, test_labels)
                accuracy_monitor.record_tensorboard(point_accuracy, iteration, sess, point_writer)
                accuracy_monitor.record_matplot(1 - point_accuracy, iteration, 'point_estimation')

                # Bayesian Estimation
                posterior_accuracy = 0
                if len(posterior_samples) >0:
                    # Inference
                    posterior_outputs = posterior_expectation(model,Loss, posterior_samples, posterior_weights, test_inputs)
                    posterior_loss = Loss.CrossEntropyLoss(posterior_outputs, test_labels)
                    loss_monitor.record_tensorboard(posterior_loss.data[0], iteration, sess, posterior_writer)
                    posterior_predictions = Loss.inference_prediction(posterior_outputs)
                    posterior_accuracy = inference_accuracy(posterior_predictions, test_labels)
                    accuracy_monitor.record_tensorboard(posterior_accuracy, iteration, sess, posterior_writer)
                    accuracy_monitor.record_matplot(1 - posterior_accuracy, iteration, 'bayesian')

                if iteration % 5000 == 0:
                    print (iteration, accuracy,point_accuracy, posterior_accuracy)
                    accuracy_monitor.save_plot_matplot(log_folder, iteration)
                #posterior_samples = []
            check_point(model, optimizer, iteration, exp_name)
            iteration = iteration + 1

           # Termination
            if iteration == num_max_iteration:
                print('It is finished')
                exit()
            idx = iteration
            if idx>0 and idx%(sample_size*sample_interval*10)==0:
                def _flatten_npyfy(posterior_samples):
                    return np.array([np.concatenate([p.cpu().numpy().ravel() for p in sample.values()]) for sample in posterior_samples])
                np.save('./saves/%s/params_%i'%(exp_name, idx//(sample_size*sample_interval)),_flatten_npyfy(posterior_samples))




if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    main(arguments)

