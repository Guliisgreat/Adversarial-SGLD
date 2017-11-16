"""gan_cifar.py


Usage:
    gan_cifar.py <src_dir> <MODE> <N_TRAIN> <DIM> <TASK>


Example:
    python gan.py CE.fc1-100-X-sgld-baby-X-babymnist@2017-11-06 wgan 10000 200 babymnist
"""

import os, sys
sys.path.append(os.getcwd())
from docopt import docopt
import time

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
# import tflib.ops.batchnorm
import tflib.plot
from tqdm import tqdm


def data_generator(src_dir, batch_size,N_TRAIN):
    f_data = filter(lambda s: s.split('.')[-1]=='npy', os.listdir(os.path.join('saves', src_dir)))
    data = np.concatenate([np.load(os.path.join(os.path.join('saves', src_dir),f_datum)) for f_datum in f_data], 0)
    data = data[:N_TRAIN]
    def get_epoch():
        # rng_state = np.random.get_state()
        np.random.shuffle(data)
        # np.random.set_state(rng_state)
        for i in xrange(len(data) / batch_size):
            yield data[i*batch_size:(i+1)*batch_size]
    return get_epoch


arguments = docopt(__doc__)
src_dir = arguments['<src_dir>']

N_TRAIN = int(arguments['<N_TRAIN>'])
TASK = arguments['<TASK>']


DATA_DIR = './data'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')
exp_dir_prefix = ''
MODE = arguments['<MODE>'] # Valid options are dcgan, wgan, or wgan-gp
DIM = int(arguments['<DIM>']) #defualt:128, This overfits substantially; you're probably better off with 64
ZDIM = 32
exp_dir = exp_dir_prefix+':'+TASK+'-'+MODE+'-'+'%g'%N_TRAIN + '%g'%DIM
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 100 # Batch size
# BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for
TESTING = False
lib.print_model_settings(locals().copy())
# Dataset iterators
train_gen = data_generator(src_dir, BATCH_SIZE, N_TRAIN=N_TRAIN)

OUTPUT_DIM = list(train_gen())[0].shape[1]

if not os.path.exists(os.path.join('gan_exps',exp_dir)):
    os.makedirs(os.path.join('gan_exps',exp_dir))



def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.relu(output)
    return output

def Generator(n_samples):
    noise = tf.random_normal([n_samples, ZDIM])
    output = ReLULayer('Generator.1', ZDIM, DIM, noise)
    output = ReLULayer('Generator.2', DIM, DIM, output)
    output = ReLULayer('Generator.3', DIM, DIM, output)
    output = lib.ops.linear.Linear('Generator.4', DIM, OUTPUT_DIM, output)
    return output

def Discriminator(inputs):
    output = ReLULayer('Discriminator.1', OUTPUT_DIM, DIM, inputs)
    output = ReLULayer('Discriminator.2', DIM, DIM, output)
    output = ReLULayer('Discriminator.3', DIM, DIM, output)
    output = lib.ops.linear.Linear('Discriminator.4', DIM, 1, output)
    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in disc_params:
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    # Standard WGAN loss
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    # Gradient penalty
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))
    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                                                  var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                                                                   var_list=lib.params_with_name('Discriminator.'))
import torch
from model.fc import *
from opt.loss import *
from torch.autograd import Variable
import itertools
import matplotlib.pyplot as plt
from collections import OrderedDict
Loss = CE()
tft = lambda x:torch.FloatTensor(x)
tfv = lambda x:Variable(tft(x))
if TASK=='toy2d':
    ## Data
    size = 10
    m1 = [-2,-2]
    m2 = [2,2]
    cov = np.eye(2) * .3
    x1 = np.random.multivariate_normal(m1, cov,size=size)
    x2 = np.random.multivariate_normal(m2, cov,size=size)
    X = np.vstack([x1,x2])
    Y = np.zeros((size*2, 2))
    Y[:size,0]=1
    Y[size:,1]=1

    X = tfv(X)
    Y = tft(Y)
    linspace = np.arange(-5,5,0.1)
    test_points = np.array(list(itertools.product(linspace, linspace)))


    def _plot(test_points, py1_xs ):
        im = np.zeros((99,99))
        for (py1_x, tp) in zip(py1_xs, test_points):
            row, col = int(tp[0]*10+50), int(tp[1]*10+50)
            im[row, col] = py1_x
        return im

    model = fc(Hn=10, input_dim=2,output_dim=2)
    model.type(torch.FloatTensor)
    def toy2d_validate(posterior_samples, idx):
        posterior_weights = [1 for _ in xrange(len(posterior_samples))]
        def _validate_batch_bayes(posterior_samples,posterior_weights, X_val_batch):
            model.eval()
            acc_proba = None
            for sample_idx in xrange(len(posterior_samples)):
                p_sample = posterior_samples[sample_idx]
                model.model.load_state_dict(p_sample)
                _,proba = Loss.infer(model, Variable(torch.FloatTensor(X_val_batch)).type(torch.FloatTensor), ret_proba=True)
                if acc_proba is None:
                    acc_proba = posterior_weights[sample_idx] * proba
                else:
                    acc_proba += posterior_weights[sample_idx] * proba
            model.train()
            acc_proba /= sum(posterior_weights)
            return acc_proba[:,0]
        bayes_probas = _validate_batch_bayes(posterior_samples,posterior_weights, test_points)
        npX = X.data.numpy()
        plt.scatter(npX[:size,0]*10+50,npX[:size,1]*10+50,c='k')
        plt.scatter(npX[size:,0]*10+50,npX[size:,1]*10+50,c='w')
        
        plt.imshow(_plot(test_points, bayes_probas), cmap=plt.cm.rainbow,interpolation='bicubic')
        plt.savefig('bayes_probas_%g.png'%idx)
elif TASK=='babymnist':
    from dataset.BabyMnist import BabyMnist
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    from train_new import posterior_expectation,inference_accuracy
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = BabyMnist( train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False, num_workers=2)
    model = fc1(100, input_dim=64)
    dataiter = iter(testloader)
    test_inputs, test_labels = dataiter.next()
    test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
    
    def babymnist_validate(posterior_samples, idx):
        posterior_weights = [1 for _ in xrange(len(posterior_samples))]
        posterior_outputs = posterior_expectation(model,Loss, posterior_samples, posterior_weights, test_inputs)
        posterior_loss = Loss.CrossEntropyLoss(posterior_outputs, test_labels)
        posterior_predictions = Loss.inference_prediction(posterior_outputs)
        posterior_accuracy = inference_accuracy(posterior_predictions, test_labels)
        lib.plot.plot('classification acc', posterior_accuracy)
else:
    raise NotImplementedError()


if TASK == 'toy2d':
    task_f = toy2d_validate
elif TASK == 'babymnist':
    task_f = babymnist_validate

def _prepare_torch_dicts(sameple_params):
    d = model.model.state_dict()
    ret = []
    for p in sameple_params:
        curr_idx = 0
        curr = OrderedDict()
        for (k,elem_d) in d.items():
            size = np.array(elem_d.size())
            curr[k] = torch.FloatTensor(p[curr_idx:curr_idx+np.prod(size)].reshape(size))
            curr_idx += np.prod(size)
        ret.append(curr)
    return ret

def inf_train_gen():
    while True:
        for images in train_gen():
            # print ('db...')
            # print (images.shape)
            yield images
saver = tf.train.Saver()

if TESTING:
    pass
# Train loop
else:
    os.chdir(os.path.join('gan_exps',exp_dir))
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        gen = inf_train_gen()

        for iteration in xrange(ITERS):
            start_time = time.time()
            # Train generator
            if iteration > 0:
                _ = session.run(gen_train_op)
            # Train critic
            if MODE == 'dcgan':
                disc_iters = 1
            else:
                disc_iters = CRITIC_ITERS
            for i in xrange(disc_iters):
                _data = gen.next()
                _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data: _data})
                if MODE == 'wgan':
                    _ = session.run(clip_disc_weights)

            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('time', time.time() - start_time)

            
            # # Calculate dev loss and generate samples every 100 iters
            # if iteration % 100 == 99:
            #     dev_disc_costs = []
            #     for images,_ in dev_gen():
            #         _dev_disc_cost = session.run(disc_cost, feed_dict={real_data_int: images}) 
            #         dev_disc_costs.append(_dev_disc_cost)
            #     lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
            #     generate_image(iteration, _data)

            # Save logs every 100 iters
            if (iteration < 5) or (iteration % 100 == 99):
                lib.plot.flush()
            lib.plot.tick()
            if (iteration > 0) and (iteration % 100 == 0):
                sameple_params = session.run(fake_data)
                task_f(_prepare_torch_dicts(sameple_params), iteration)
