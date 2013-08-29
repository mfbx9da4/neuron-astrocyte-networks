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
from collections import namedtuple

import matplotlib.pyplot as plt
from numpy import array
import numpy as np

from plotting.plotters import plotPercentageErrorBar, plotPercentageNoErrorBar



BarVals = namedtuple('BarVals', 'BNN PPM PM')
trn_means = BarVals(7.26, 11, 20)
trn_se =   BarVals(5.9/np.sqrt(20.), .2, .4)
tst_means = BarVals(1.3, 14, 25)
tst_se =   BarVals(0., .1, 0.2)

def drawBar():
    N = 3


    ind = np.arange(N)  # the x locations for the groups
    width = .4       # the width of the bars

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    trn_bars = ax.bar(ind, trn_means, width, color='#3366CC', ecolor='#000000', yerr=trn_se)
    tst_bars = ax.bar(ind+width, tst_means, width, color='#DC3912', ecolor='#000000', yerr=tst_se)

    # add some
    ax.set_ylabel('Percentage error')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(('BNN', 'PPM', 'RAP'))

    ax.legend((trn_bars[0], tst_bars[0]), ('Final training', 'Test'), loc='upper left')

    rect = trn_bars[2]
    height = rect.get_height()
    ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '**',
                    ha='center', va='bottom')

    plt.show()


if __name__ == '__main__':
    drawBar()
