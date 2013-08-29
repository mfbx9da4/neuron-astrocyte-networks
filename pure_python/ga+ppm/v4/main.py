# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:23:57 2013

Unadultered genetic algorithm

@author: david
"""
import os
from operator import itemgetter
import datetime

import matplotlib.pyplot as plt
import numpy as np
from pylab import array

from neuralnetwork import NN
from GA import createPop, pairPop, rankPop, evolveNewPop

max_iterations = 3

def testFittestInd(rankedPop):
    tester = NN ()
    fittestWeights = rankedPop[0][0]
    tester.assignWeights(fittestWeights)
    return tester.sumErrors(test_data=True)


def runTrial():
    """
    Fresh population of NN individuals are evalutated, paired with 
    their fitness scores and ordered by fitness in descending order (fittest first)
    """
    pop = createPop()
    pairedPop = pairPop(pop)
    rankedPop = sorted(pairedPop, key=itemgetter(-1), reverse=True) 
    all_results = {
        'train_mses_top' : [],
        'train_mses_avg' : [],
        'train_percents_avg' : [],
        'train_percents_top' : [],
        'test_mse' : 0,
        'test_percent' : 0
    }
    for iters in range(max_iterations):
        if not iters % 1:
            print 'Iteration'.rjust(150), iters
        newpopW = evolveNewPop(rankedPop)
        rankedPop, top_mse, avg_mse, top_per, avg_per = rankPop(newpopW, pop)
        all_results['train_mses_top'].append(top_mse)
        all_results['train_mses_avg'].append(avg_mse)
        all_results['train_percents_top'].append(top_per)
        all_results['train_percents_avg'].append(avg_per)
    test_mse, test_per = testFittestInd(rankedPop)
    all_results['test_mse'] = test_mse
    all_results['test_percent'] = test_per
    return all_results


def main (trials=3):
    all_trials = {
        'train_mses_top' : [],
        'train_mses_avg' : [],
        'train_percents_avg' : [],
        'train_percents_top' : [],
        'test_mse' : [],
        'test_percent' : []
    }
    for i in range(trials):
        trial_results = runTrial()
        for k in trial_results.keys():
            all_trials[k].append(trial_results[k])

    a = datetime.datetime.now().    utctimetuple()
    time_string = str(a[3]) + str(a[4]) + '_' + str(a[2]) + '-' + \
        str(a[1]) + '-' + str(a[0])
    if os.environ['OS'] == 'Windows_NT':
        sep = '\\'
    else:
        sep = '/'
    cur_dir = os.getcwd() + sep
    os.mkdir(cur_dir + time_string)
    for k in all_trials.keys():
        f = open(cur_dir + time_string + sep + k + '.out', 'w')
        np.savetxt(f, all_trials[k])
    
    implementation_parameters = {
    'trials' : trials,
    'max_iterations' : max_iterations,
    'pop_size' : NN.pop_size,
    'elitism_individuals_num' : NN.eliteN,
    'mutation_rate' : NN.mutation_rate,
    'crossover_rate' : NN.crossover_rate,
    'NN_indim' : NN.ni,
    'NN_hiddim' : NN.ni,
    'NN_outdim' : NN.no
    }
    f = open(cur_dir + time_string + sep + 
        'implementation_parameters.out', 'w')
    f.writelines(str(implementation_parameters.items()))
    f.close()


  
if __name__ == "__main__":
    main()

