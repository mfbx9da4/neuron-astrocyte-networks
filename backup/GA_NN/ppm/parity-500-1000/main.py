# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:23:57 2013

Unadultered genetic algorithm

@author: david
"""
import sys
import os
from operator import itemgetter
import datetime

import matplotlib.pyplot as plt
import numpy as np
from pylab import array


from neuralnetwork import NN
from GA import createPop, pairPop, rankPop, evolveNewPop, trainPP

max_iterations = 1000
trials = 20

def testFittestInd(rankedPop):
    tester = NN ()
    fittestWeights = rankedPop[0][0]
    tester.assignWeights(fittestWeights)
    right = 0
    for i, t in tester.test_pat:
        o = tester.activate(i)
        print i, o, t, '++' if list(o) == list(t) else '----'
        right += 1 if list(o) == list(t) else 0
    print right, len(tester.test_pat)
    print 100 * (float(right) / len(tester.test_pat))
    err, per = tester.sumErrors(test_data=True)
    print per
    return err, per


def runTrial(iters):
    """
    Fresh population of NN individuals are evalutated, paired with 
    their fitness scores and ordered by fitness in descending order (fittest first)
    """
    pop = createPop()
    pairedPop = pairPop(pop, verbose=False)
    rankedPop = sorted(pairedPop, key=itemgetter(-1), reverse=True) 
    all_results = {
        'train_mses_top' : [],
        'train_mses_avg' : [],
        'train_percents_avg' : [],
        'train_percents_top' : [],
        'test_mse' : 0,
        'test_percent' : 0
    }
    for iters in range(iters):
        # raw_input('==============NEXT==============')
        if not iters % 100:
            print 'Iteration'.rjust(120), iters
        newpopW = evolveNewPop(rankedPop)
        # print '\n \nEVOLVE NEW POP'
        rankedPop, top_mse, avg_mse, top_per, avg_per = rankPop(newpopW, pop, verbose=False)
        if iters > 500:
            newpopW = trainPP(rankedPop, pop, incr=0.25, decr=0.5)
            # print '\n \nTRAIN PP'
            rankedPop, top_mse, avg_mse, top_per, avg_per = rankPop(newpopW, pop, verbose=False)
        all_results['train_mses_top'].append(top_mse)
        all_results['train_mses_avg'].append(avg_mse)
        all_results['train_percents_top'].append(top_per)
        all_results['train_percents_avg'].append(avg_per)
    print top_per   
    test_mse, test_per = testFittestInd(rankedPop)
    all_results['test_mse'] = test_mse
    all_results['test_percent'] = test_per
    return all_results


def main ():
    all_trials = {
        'train_mses_top' : [],
        'train_mses_avg' : [],
        'train_percents_avg' : [],
        'train_percents_top' : [],
        'test_mse' : [],
        'test_percent' : []
    }
    trials, iters = getArgs()
    for i in range(trials):
        trial_results = runTrial(iters)
        for k in trial_results.keys():
            all_trials[k].append(trial_results[k])

    a = datetime.datetime.now().utctimetuple()
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

def getArgs():
    if len(sys.argv) > 1:
        if len(sys.argv) > 2:
            trials = int(sys.argv[1])
            iters = int(sys.argv[2])
        else:
            iters = int(sys.argv[1])
    else: 
        trials = 1
        iters = 4
    return trials, iters

  
if __name__ == "__main__":
    main()

