# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:51:23 2013

General plotting

@author: david
"""
from pylab import *
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

def plotPercentageErrorBar(data):
    means = np.mean(data, axis=0)
    std_errors = np.std(data, axis=0)/np.sqrt(len(data))
    plt.errorbar(range(len(means)), means, yerr=std_errors)
    plt.ylim(0,100)
    plt.ylabel('Percentage error (%)')
    print 'means', means
    print 'std_errors', std_errors

def plotPercentageNoErrorBar(data, label='', xytxt=(0.8,1.02)):
    data = array(data)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)/np.sqrt(len(data))
    plt.plot(range(len(means)), means)
    plt.ylim(0,100)
    plt.ylabel('Percentage error (%)')
    plt.xlabel('iterations')
    txt = label+'$\mu$='+str(means[-1])+'\n$\sigma$='+str(stds[-1])
    plt.annotate(txt, xy=(0.5,0.5), xytext=xytxt, xycoords='axes fraction',
           textcoords='axes fraction', fontsize='11')

def plotErrorsMeansErrorBar(data, ylimit=False):
    data = data['errors']
    means = np.mean(data, axis=0)
    std_errors = np.std(data, axis=0)/np.sqrt(len(data))
    errorbar(range(len(means)), means, yerr=std_errors)
    if ylimit:
        ylim(ylimit)


def plotErrorsMeansNoErrorBar(data, label='', xytxt=(0.8,1.02), 
        ylimit=False, showFinalValues=True):
    data = data['errors']                            
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)/np.sqrt(len(data))
    plot(range(len(means)), means)
    if ylimit:
        ylim(ylimit)
    if showFinalValues:
        txt = label+'$\mu$='+str(means[-1])+'\n$\sigma$='+str(stds[-1])
        annotate(txt, xy=(0.5,0.5), xytext=xytxt, xycoords='axes fraction',
                 textcoords='axes fraction', fontsize='11')


def plotErrorsAllTrials(data):
    for trial in data['errors']:
        plot(trial)

def plotBar(data):
    """
    bar([0.25, 0.75], map(int, rand(2)*10), yerr=[1, 2], width=0.5)
    first arg is postions of bar 
    width is width of bar
    
    a = axes()
    a.xaxis.set_label_text('x label')
    a.xaxis.set_ticklabels(['', 'bp', '', 'angn'])
    
    """

def plotWeightsAllTrials(data):
    """allWeights contains arrays - each of which is a trial
    Each trial should contain arrays each containing values 
    of said weight for each iteration
    
    q is an approximate for the grid
    """
    allWeights = data['weights']
    allErrors = data['errors']
    trial = 0
    for trialVals in allWeights:
        assert array(trialVals).shape[1] == len(allErrors[0]), 'vals not transposd %s' % str(array(trialVals).shape)
        params_n = len(trialVals)
        q = int(sqrt(params_n))+1
        trialError = allErrors[trial][-1]
        for i in range(params_n):
            s = subplot(q, q-1, i+1)
            s.axes.get_xaxis().set_visible(False)
            if i == params_n-1:
                s.axes.get_xaxis().set_visible(True)    
#                s.xaxis.set_major_locator(MultipleLocator(20))
            s.axes.get_yaxis().set_visible(False)
            if i == 0:
                s.axes.get_yaxis().set_visible(True)
                s.yaxis.set_major_locator(MultipleLocator(2))
            subplots_adjust(hspace=0.1, wspace=0.1)
            plot(trialVals[i], label=str(trialError))
            ylim(-4,4)
        trial+=1
    plt.legend(bbox_to_anchor=(1.5, 1), prop={'size':8})


def plotActivationsAllTrials(data, col_lengths):
    """
    key:
        i: current_neuron
    """
    allActivations = data['activations']
    allErrors = data['errors']
    assert array(allActivations).shape[2] == array(allErrors).shape[1], 'vals not transposd correctly %s' % str(array(trialVals).shape)
    plotBools = convertToBoolean(col_lengths)
    cols = max(col_lengths)
    rows = len(col_lengths)
    params_n = array(allActivations).shape[1]
    print array(allActivations).shape
    i = 0
    print plotBools
    for current_plot, isPlot in enumerate(plotBools):
        if isPlot:
            current_plot+=1
            print i, current_plot
            subplot(rows, cols, current_plot)
            subplots_adjust(hspace=0.1, wspace=0.1)
            for triali, trialVals in enumerate(allActivations):
                annotate(str(i)+' : '+str(current_plot), xy=(0.5, 0.5), xycoords='axes fraction')
                plot(trialVals[i], label=str(allErrors[triali][-1]))
            i += 1
        

def convertToBoolean(col_lengths):
    c = []
    for x in col_lengths:
        c.extend(x*[True])
        c.extend((max(col_lengths)-x)*[False])
    return c

