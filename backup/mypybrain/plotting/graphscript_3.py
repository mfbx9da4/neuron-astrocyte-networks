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

from plotting.plotters import plotPercentageErrorBar, plotPercentageNoErrorBar


def main():
    args = sys.argv[1:]

    if not args:
        print 'usage: [--path] path [--algo] name'
        sys.exit(1)

    path = args[0]
    f = open(path + 'train_mses_top.out', 'r')
    train_mses = np.loadtxt(f)
    f = open(path + 'train_percents_top.out', 'r')
    train_percs = np.loadtxt(f)
    
    print train_mses
    print train_percs
    
    shape = array(train_mses).shape
    try:
        repeats, iterations = shape
    except ValueError:
        iterations = shape[0]
        repeats = 1
    # f = plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plotPercentageErrorBar(train_percs)
    # plt.subplot(1, 2, 2)
    # side_text = 'Trials=' + str(repeats) + '\n\nFinal\ntrain\nerror\n'
    # plotPercentageNoErrorBar(train_percs, label=side_text,
    #     xytxt=(1.02, 1.3))
    # plotPercentageNoErrorBar(train_percs, label='Final\ntest\nerror\n',
    #     xytxt=(1.02, 0.4))
    # f.savefig(path + '_bp-Threshold-Layer' + '_means.png', format='png')

    f = plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title('MSEs')
    # for trial in train_mses:
    #     plt.plot(trial)
    # plt.subplot(1, 2, 2)
    # plt.title('Percents')
    # for trial in train_percs:
    #     plt.plot(trial)
    plt.plot(train_percs)
    plt.ylim(0,1)
    f.savefig(path + '_bp-Threshold-Layer' + '_all_trials.png', format='png')



if __name__ == '__main__':
    main()
