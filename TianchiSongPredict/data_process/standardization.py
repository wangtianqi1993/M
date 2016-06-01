# !/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'wtq'

from numpy import *


def standardization(x_mat, y_mat=[1,2,3,4]):
    """
    所有的特征减去各自的均值并除以方差,使每维度特征有相同的重要性
    :return:
    """
    x_means = mean(x_mat, 0)
    x_var = var(x_mat, 0)
    x_var = sqrt(x_var)
    x_mat = (x_mat - x_means)/x_var
    return x_mat, y_mat

if __name__ == '__main__':
    a = [[1,2,3,4],
         [2,3,4,5]]
    b, w = standardization(a)
    print b[0]
