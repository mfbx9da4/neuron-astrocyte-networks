# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:51:23 2013

General plotting

@author: david
"""
from pylab import *

def plotMeansErrorBar(data, ylimit=False):
    figure()
    means = np.mean(data, axis=0)
    std_errors = np.std(data, axis=0)/np.sqrt(len(data))
    errorbar(range(len(means)), means, yerr=std_errors)
    if ylimit:
        ylim(ylimit)
    show()


def plotMeansNoErrorBar(data, label='', xytxt=(0.8,1.02), ylimit=False,
                        showFinalValues=True):
    figure()
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)/np.sqrt(len(data))
    plot(range(len(means)), means)
    if ylimit:
        ylim(ylimit)
    if showFinalValues:
        txt = label+'$\mu$='+str(means[-1])+'\n$\sigma$='+str(stds[-1])
        annotate(txt, xy=(0.5,0.5), xytext=xytxt, xycoords='axes fraction',
                 textcoords='axes fraction', fontsize='11')
    show()

def plotAllTrials(data):
    figure()
    for trial in data:
        plot(trial)
    show()
           

#data = rand(300).reshape(15, 20)
#plotErrorBar(data)
#plotNoErrorBar(data)
#show()