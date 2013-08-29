# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 01:04:42 2013

@author: david
"""

import random
import datetime
import os
import sys
path = os.getcwd()
i = path.index('mypybrain')
path = path[:i + 10]
sys.path.append(path)

import matplotlib.pyplot as plt
from numpy import array
import numpy as np


def main():
    args = sys.argv[1:]

    if not args:
        print 'usage: [--path] path [--algo] name'
        sys.exit(1)

    path = args[0]
    file_name = 'trn_per_err'
    try:
        f = open(path + file_name + '.out', 'r')
        data = np.loadtxt(f)
    except IOError:
        f = open(path + 'train_percents_top' + '.out', 'r')
        data = np.loadtxt(f)
        for i, trial in enumerate(data):
            data[i][:] = 100 - data[i]


    
    f = plt.figure()
    data = array(data)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)/np.sqrt(len(data))
    plt.errorbar(range(len(means)), means, yerr=stds)
    plt.ylim(0,100)
    # plt.ylabel('Percentage error (%)')
    # plt.xlabel('Iterations')
    f.savefig(path + file_name + '.png', format='png')



if __name__ == '__main__':
    main()
