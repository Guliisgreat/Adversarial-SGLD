from __future__ import print_function
from sklearn import datasets, svm, metrics
import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs



class BabyMnist(data.Dataset):


    def __init__(self,  train=True, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.data = datasets.load_digits()
        self.image = self.data.images.reshape((-1,8,8,1))
        self.image = self.image[0:1701,:,:,:]  # avoid conflict with batch size
        self.label = self.data.target
        self.label = self.label[0:1701]
        self.size_test_dataset = 100


        if self.train:
            self.train_data, self.train_labels = self.image[self.size_test_dataset:-1,:,:,:], self.label[self.size_test_dataset:-1]
        else:
            self.test_data, self.test_labels = self.image[:self.size_test_dataset,:,:,:], self.label[:self.size_test_dataset]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)



