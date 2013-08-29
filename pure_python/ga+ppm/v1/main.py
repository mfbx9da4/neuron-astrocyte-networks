# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:23:57 2013

@author: david
"""
from operator import itemgetter
import GADANN_test as tst
import matplotlib.pyplot as plt
import numpy as np
from neuralnet import NN


graphical_error_scale = 100
max_iterations = 1000
pop_size = 100
mutation_rate = 0.1
crossover_rate = 0.8
ni, nh, no = 4,6,1



def main ():
  """
  Fresh population of NN individuals are evalutated, paired with 
  their fitness scores and ordered by fitness in descending order (fittest first)
  """
  # Rank first random population
  pop = [ NN(ni,nh,no) for i in range(pop_size) ] # fresh pop
  pairedPop = pairPop(pop)
  # ordered by fitness in descending order
  rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True) 
  tst.correctlyranked(rankedPop)
  # Keep iterating new pops until max_iterations
  iters = 0
  tops, avgs = [], []
  while iters != max_iterations:
    if iters%1 == 0:
      print 'Iteration'.rjust(150), iters
    newpopW = iteratePop(rankedPop)
    rankedPop, toperr, avgerr = rankPop(newpopW,pop)
    tops.append(toperr)
    avgs.append(avgerr)
    iters+=1
  
  # test a NN with the fittest weights
  tester = NN (ni,nh,no)
  fittestWeights = [ x[0] for x in rankedPop ]
  tester.assignWeights(fittestWeights, 0)
  results, targets = tester.test(testpat)
  x = np.arange(0,150)
  title2 = 'Test after '+str(iters)+' iterations'
  plt.title(title2)
  plt.ylabel('Node output')
  plt.xlabel('Instances')
  plt.plot( results, 'xr', linewidth = 0.5)
  plt.plot( targets, 's', color = 'black',linewidth = 3)
  #lines = plt.plot( results, 'sg')
  plt.annotate(s='Target Values', xy = (110, 0),color = 'black', family = 'sans-serif', size  ='small')
  plt.annotate(s='Test Values', xy = (110, 0.5),color = 'red', family = 'sans-serif', size  ='small', weight = 'bold')
  plt.figure(2)
  plt.subplot(121)
  plt.title('Top individual error evolution')
  plt.ylabel('Inverse error')
  plt.xlabel('Iterations')
  plt.plot( tops, '-g', linewidth = 1)
  plt.subplot(122)
  plt.plot( avgs, '-g', linewidth = 1)
  plt.title('Population average error evolution')
  plt.ylabel('Inverse error')
  plt.xlabel('Iterations')
  
  plt.show()
  
  print 'max_iterations',max_iterations,'\tpop_size',pop_size,'pop_size*0.15',int(pop_size*0.15),'\tmutation_rate',mutation_rate,'crossover_rate',crossover_rate,'ni, nh, no',ni, nh, no



  
if __name__ == "__main__":
    main()

