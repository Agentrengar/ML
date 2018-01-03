#!usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


class Perceptron(object):
    """Perceptron classifier.
    Parameters
    ------------
    n_rate : float, Learning rate(between 0,0 and 1.0)
    n_iter : int, Passes over the training dataset.

    Attributes
    ------------
    w_n : 1d-array, Weights after fitting
    errors_n :list, Number of misclassifications in every epoch.
    """
    def __init__(self, n_rate=0.01, n_iter=10):
        self.n_rate = n_rate
        self.n_iter = n_iter

    def fit(self, x, y):
        """Fit training data.先对权重参数初始化，然后对训练集中得每一个参数进行初始化，根据感知机算法规则对群众进行更新
        :param x: array-like, shape = [n_samples, n_features], Training vectors, where n_samples is the number of sampl
        es and n_features is the number of features.
        :param y: array-like, shape = [n_samples], Target values.
        :return: self: object
        """
        self.w_n = np.zeros(1 + x.shape[1])
