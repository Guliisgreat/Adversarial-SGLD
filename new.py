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

import datetime
from opt.loss import *
from model.fc import fc, lr
from opt.nsgd import NoisedSGD
import torch.optim as optim

# from model.cnn import *
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from tensorboard_monitor.configuration import*
from tensorboard_monitor.monitor import*


def load_configuration(arguments):
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
    if not os.path.exists('./saves/%s' % exp_name):
        os.makedirs('./saves/%s' % exp_name)

    ## Model
    model = eval(model_config['name'])(**model_config['kwargs'])
    # model.type(torch.FloatTensor)
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

# def prediction_accuracy(outputs, labels):
#     prediction = outputs.data.numpy().argmax(1)
#     accuracy = (prediction == labels.data.numpy()).mean().astype(float)
#     return accuracy
def inference_accuracy(prediction, labels):
    accuracy = (prediction == labels.data.numpy()).mean().astype(float)
    return accuracy


def posterior_expectation(model,Loss, posterior_samples, posterior_weights, inputs ):
    num_posterior_samples = len(posterior_samples)
    outputs_weighted_sum = 0
    # we need to load posterior parameter samples to make inference, but cannot influence the training process
    copy_model = copy.deepcopy(model)
    for sample_idx in range(num_posterior_samples):
        copy_model.model.load_state_dict(posterior_samples[sample_idx])
        outputs_samples =  copy_model.forward(inputs)
        outputs_prob = Loss.softmax_output(outputs_samples)
        # outputs_sample --> softmax --> prob
        outputs_weighted_sum = outputs_weighted_sum + outputs_prob*posterior_weights[sample_idx]
    outputs_expectation = outputs_weighted_sum / (sum(posterior_weights))
    return outputs_expectation

def is_into_langevin_dynamics(model,outputs, gradient_data, optimizer, alpha_threshold, learning_rate, batch_size):
    is_collecting = False
    gs = []
    oG = gradient_data.type(torch.FloatTensor)
    for gidx in xrange(oG.size()[0]):
        model.zero_grad()
        outputs.backward(gradient=oG[gidx][None].repeat(batch_size, 1), retain_variables=True)
        ps = list(model.parameters())
        g = torch.cat([p.grad.view(-1, 1) for p in ps])
        gs.append(g)
    gs = torch.cat(gs, 1).t()  ## (N, D) gradient matrix
    gs_centered = gs - gs.mean(0).expand_as(gs)
    V_s = (1 / optimizer.correction) * torch.mm(gs_centered, gs_centered.t())  ## (D, D), hopefully V_s in the paper
    _, s, _ = torch.svd(V_s.data)
    alpha = s[0] * learning_rate * optimizer.correction / batch_size

    if alpha < alpha_threshold:  # todo: ...
        is_collecting = True
    return is_collecting, s[0]

def posterior_sampling(sample_size, model,learning_rate, posterior_samples, posterior_weights):
    posterior_samples.append(copy.deepcopy(model.model.state_dict()))
    posterior_weights.append(learning_rate)
    if len(posterior_samples) > sample_size:
        del posterior_samples[0]
        del posterior_weights[0]
    return posterior_samples, posterior_weights

# def store_gradient_data(outputs, criterion, label):
#     outputs = outputs.data.clone().type(torch.FloatTensor)
#     outputs = Variable(outputs, requires_grad=True)
#     loss = criterion(outputs, label)
#     loss.backward()
#     gradient_data = outputs.grad.data
#     return gradient_data




def main(arguments):

    # Set up
    iteration = 0
    batch_size = 100
    alpha_threshold = 0.01
    sample_size = 100
    sample_interval = 20
    posterior_samples = []
    posterior_weights = []
    validation_interval = 200
    variance_monitor_interval = 50

    # Load argumentts: Module, Optimizer, Loss_function
    model, optimizer, Loss, exp_name, model_config, opt_config = load_configuration(arguments)
    num_max_iteration = opt_config['max_train_iters']

    criterion = nn.CrossEntropyLoss()


  # Load DataSet
    # trainLoader automatically generate training_batch
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # size_dataset = np.size(trainset.train_data.numpy())
    # print(size_dataset)
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
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            # update learning rate
            learning_rate = update_LearningRate(optimizer, iteration, opt_config)
            learning_rate_monitor.record_tensorboard(learning_rate,iteration,sess,train_writer)
            # Inference
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            gradient_data =  Loss.store_gradient_data(outputs,labels)

            # gradient_data = store_gradient_data(outputs, criterion, labels)
            # Loss function
            training_loss = Loss.CrossEntropyLoss(outputs,labels)
            loss_monitor.record_tensorboard(training_loss.data[0],iteration,sess,train_writer)
            # loss = criterion(outputs, labels)
            # accuracy = prediction_accuracy(outputs, labels)
            training_predictions = Loss.inference_prediction(outputs)
            accuracy = inference_accuracy(training_predictions, labels)
            accuracy_monitor.record_tensorboard(accuracy, iteration,sess, train_writer)
            # Parameter Update
            training_loss.backward(retain_variables=True)
            optimizer.step()

            # monitor Variance
            if opt_config['name'] == 'NoisedSGD' and iteration % variance_monitor_interval == 0 :
                is_collecting, V_s = is_into_langevin_dynamics(model, outputs,  gradient_data, optimizer, alpha_threshold, learning_rate, batch_size)
                variance_monitor.record_tensorboard(V_s, iteration,sess, train_writer)
                if is_collecting == True and iteration %sample_interval == 0:
                    posterior_samples, posterior_weights = posterior_sampling(sample_size, model, learning_rate, posterior_samples,
                                                                              posterior_weights)
           # Validation
            # Point Estimation
            if iteration % validation_interval == 0:
                # Load data
                dataiter = iter(testloader)
                test_inputs, test_labels = dataiter.next()
                test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
                # Inference
                point_outputs = model.forward(test_inputs)
                point_loss = Loss.CrossEntropyLoss(point_outputs, test_labels)
                loss_monitor.record_tensorboard(point_loss.data[0], iteration, sess, point_writer)
                point_predictions = Loss.inference_prediction(point_outputs)
                point_accuracy = inference_accuracy(point_predictions, test_labels)
                accuracy_monitor.record_tensorboard(point_accuracy, iteration, sess, point_writer)
                # point_loss = criterion(point_outputs, test_labels)
                # loss_monitor.record_tensorboard(point_loss.data[0], iteration, sess, point_writer)
                # point_accuracy = prediction_accuracy(point_outputs, test_labels)
                # accuracy_monitor.record_tensorboard(point_accuracy, iteration, sess, point_writer)
            # Bayesian Estimation
            if (iteration % validation_interval == 0) and (len(posterior_samples) == sample_size):
                # Inference

                posterior_outputs = posterior_expectation(model,Loss, posterior_samples, posterior_weights, test_inputs)
                posterior_loss = Loss.CrossEntropyLoss(posterior_outputs, test_labels)
                loss_monitor.record_tensorboard(posterior_loss.data[0], iteration, sess, posterior_writer)
                posterior_predictions = Loss.inference_prediction(posterior_outputs)
                posterior_accuracy = inference_accuracy(posterior_predictions, test_labels)
                accuracy_monitor.record_tensorboard(posterior_accuracy, iteration, sess, posterior_writer)
                # posterior_loss = criterion(posterior_outputs, test_labels)
                # loss_monitor.record_tensorboard(posterior_loss.data[0], iteration, sess, posterior_writer)
                # posterior_accuracy = prediction_accuracy(posterior_outputs, test_labels)
                # accuracy_monitor.record_tensorboard(posterior_accuracy, iteration, sess, posterior_writer)

            check_point(model, optimizer, iteration, exp_name)
            iteration = iteration + 1

            # Termination
            if iteration == num_max_iteration:
                print('It is finished')
                exit()




if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    main(arguments)

