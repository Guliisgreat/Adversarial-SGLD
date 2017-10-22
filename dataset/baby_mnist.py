# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def baby_mnist():
    size_test = 170
# The digits dataset
    digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.

# image_data: (num, 8, 8, 1)
    images_and_labels = list(zip(digits.images, digits.target))
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples,8,8,1))
# one hot label
    label = digits.target
    onehot_encoder = OneHotEncoder(sparse=False)
    label = label.reshape(len(label), 1)
    label = onehot_encoder.fit_transform(label)


    test_data = data[:size_test,:,:,:]
    test_label = label[:size_test]
    train_data = data[size_test:-1,:,:,:]
    train_label = label[size_test:-1]

    return train_data, train_label, test_data, test_label