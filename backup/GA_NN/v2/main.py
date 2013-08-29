# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:23:57 2013

@author: david
"""
from operator import itemgetter
# import GADANN_test as tst
import matplotlib.pyplot as plt
import numpy as np
from neuralnetwork import NN
from GA import createPop, pairPop, rankPop, evolveNewPop
from iris import testpat



# graphical_error_scale = 100
max_iterations = 10
# pop_size = 150
# mutation_rate = 0.1
# crossover_rate = 0.8
# ni, nh, no = 4,6,1



def main ():
    """
    Fresh population of NN individuals are evalutated, paired with 
    their fitness scores and ordered by fitness in descending order (fittest first)
    """
    # Rank first random population
    pop = createPop()
    pairedPop = pairPop(pop)
    # ordered by fitness in descending order
    rankedPop = sorted(pairedPop, key=itemgetter(-1), reverse=True) 
    # tst.correctlyranked(rankedPop)
    # Keep iterating new pops until max_iterations
    iters = 0
    tops, avgs = [], []
    while iters != max_iterations:
        if iters%1 == 0:
            print 'Iteration'.rjust(150), iters
        newpopW = evolveNewPop(rankedPop)
        rankedPop, toperr, avgerr = rankPop(newpopW, pop)
        tops.append(toperr)
        avgs.append(avgerr)
        iters += 1
    
    # test a NN with the fittest weights
    tester = NN ()
    fittestWeights = rankedPop[0][0]
    tester.assignWeights(fittestWeights)
    results, targets = tester.test(testpat)
    title2 = 'Test after '+str(iters)+' iterations'
    plt.title(title2)
    plt.ylabel('Node output')
    plt.xlabel('Instances')
    plt.plot(results, 'xr', linewidth=1.5, label='Results')
    plt.plot(targets, 's', color='black', linewidth=3, label='Targets')
    plt.legend(loc='lower right')

    plt.figure(2)
    plt.subplot(121)
    plt.title('Top individual error evolution')
    plt.ylabel('Inverse error')
    plt.xlabel('Iterations')
    plt.plot( tops, '-g', linewidth=1)
    plt.subplot(122)
    plt.plot( avgs, '-g', linewidth=1)
    plt.title('Population average error evolution')
    plt.ylabel('Inverse error')
    plt.xlabel('Iterations')
    
    plt.show()
    
    print 'max_iterations',max_iterations,'\tpop_size', \
        NN.pop_size,'pop_size*0.15', \
        int(NN.pop_size*0.15),'\tmutation_rate', \
        NN.mutation_rate,'crossover_rate', \
        NN.crossover_rate,'ni, nh, no', NN.ni, NN.nh, NN.no



  
if __name__ == "__main__":
    main()

