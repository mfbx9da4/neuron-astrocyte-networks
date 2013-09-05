import sys
from matplotlib.pyplot import *
from math import *
from numpy import arange, array
import numpy as np
import random
from random import random as r
from NeuralNetworkObject import NN
from iris import randomdata

#pat = [[[0,0], [0]],[[0,1], [1]],[[1,0], [1]],[[1,1], [0]]];testpat = [[[0,0], [0]],[[0,1], [1]],[[1,0], [1]],[[1,1], [0]]]
trials = 10

def main ():
  ngatrainerrs, ngatrainwngs, bptrainaccs, ngatrainaccs, bptrainerrs, bptrainwngs = [],[],[],[],[],[]
  ngatesterrs, ngatestwngs, bptestaccs, ngatestaccs, bptesterrs, bptestwngs = [],[],[],[],[],[]
  isNGA = False
  for T in range(trials):
    pat, testpat = randomdata()    
    myNN = NN(pat, testpat)
    err1, wng1, acc1 = myNN.train(isNGA)
    ngatrainerrs.append(err1); ngatrainwngs.append(wng1); ngatrainaccs.append(acc1)
    err2, wng2, acc2 = myNN.test()
    ngatesterrs.append(err2); ngatestwngs.append(wng2); ngatestaccs.append(acc2)
    print T, acc2, '% accuracy'
    print array(ngatrainerrs).shape

  fig = figure(); suptitle(sys.argv[0])
  subplot(2,3,1); plot_training_data(ngatrainerrs, Title='Training error')
  subplot(2,3,2); plot_training_data(ngatrainwngs, Title='Training wrongs');xlabel('Iterations',style='italic')
  subplot(2,3,3); plot_training_data(ngatrainaccs, Title='Training accuracies (%)', isPercentage=1)
  
  subplot(2,3,4); plot_test_data(ngatesterrs,'Error') 
  subplot(2,3,5); plot_test_data(ngatestwngs,'Wrongs'); xlabel('Trials', style='italic') 
  subplot(2,3,6); plot_test_data(ngatestaccs,'Accuracies')
  show()
  return fig
  
def plot_training_data(data,Title='',isPercentage=0):
  data = array(data)
  isNGA = len(data[0][1])
  
  # during each iteration the performance is measured once after the bp and again after the nga
  data_after_bp = data[:,0,:]
  bp_averaged_values = np.mean(data_after_bp, axis=0)
  bp_std_errors = np.std(data_after_bp, axis=0)/sqrt(len(data_after_bp))
  errorbar(range(len(bp_averaged_values)),bp_averaged_values, \
          yerr=bp_std_errors, fmt='b-', label='Post BP')
          
  if isNGA:
    data_after_nga = data[:,1,:]
    assert len(data_after_bp)==len(data_after_nga), 'length of data wrong'
    nga_averaged_values = np.mean(data_after_nga, axis=0)
    nga_std_errors= np.std(data_after_nga, axis=0)/sqrt(len(data_after_nga))
    errorbar(range(len(nga_averaged_values)),nga_averaged_values, \
          yerr=nga_std_errors, fmt='g-', label='Post NGA')
  
  title(Title)
  if isPercentage: ylim((0,110))

def plot_test_data(data,datatype):
  plot(range(trials),data); title('Test '+datatype)
  mean = round(np.mean(data), 2)
  std = round(np.std(data), 2)
  txt = '$\mu$='+str(mean)+'\n$\sigma$='+str(std)
  ylim((-0.01, max(data)*1.01))
  annotate(txt, xy=(0.5,0.5), xytext=(0.8,1.02), xycoords='axes fraction', \
          textcoords='axes fraction', fontsize='11')

  
if __name__ == "__main__":
    main()
