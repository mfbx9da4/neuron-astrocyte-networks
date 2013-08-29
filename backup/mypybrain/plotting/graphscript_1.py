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
    f = open(path + 'all_trn_results.out', 'r')
    all_trn_results = np.loadtxt(f)
    f = open(path + 'all_tst_results.out', 'r')
    all_tst_results = np.loadtxt(f)
    print all_trn_results
    print all_tst_results
    repeats, iterations = array(all_trn_results).shape
    f = plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plotPercentageErrorBar(all_trn_results)
    plotPercentageErrorBar(all_tst_results)
    plt.subplot(2, 1, 2)
    side_text = 'Trials=' + str(repeats) + '\n\nFinal\ntrain\nerror\n'
    plotPercentageNoErrorBar(all_trn_results, label=side_text,
        xytxt=(1.02, 1.3))
    plotPercentageNoErrorBar(all_tst_results, label='Final\ntest\nerror\n',
        xytxt=(1.02, 0.4))
    plt.legend(('Training', 'Test'))
    f.savefig(path + '_bp-Threshold-Layer' + '_means.png', format='png')

    f = plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.title('Training')
    for trial in all_trn_results:
        plt.plot(trial)
    plt.subplot(2, 1, 2)
    plt.title('Test')
    for trial in all_tst_results:
        plt.plot(trial)
    f.savefig(path + '_bp-Threshold-Layer' + '_all_trials.png', format='png')



if __name__ == '__main__':
    main()
