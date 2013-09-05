import random
import sys
import os
if os.environ['OS'] == 'Windows_NT':
    sys.path.append('C:\\Users\\david\\Dropbox\\programming\\python\\ann\\myangn\\sem6')
    sys.path.append('C:\\Users\\david\\Dropbox\\programming\\python\\ann\\mypybrain')
else:
    sys.path.append('/home/david/Dropbox/programming/python/ann/myangn/sem6')
    sys.path.append('home/david/Dropbox/programming/python/ann/mypybrain')

from pylab import array, ones, zeros, where, rand
import numpy as np
from pybrain.utilities import percentError

from utilities import percentAcc, OutToClass

# to do:
#   * change iris to be 3d data
#   * make sure data is presented randomly each time
#   * turn into 3d output
#   * return percentage error
#   * implement PP
#   * implement my algo



class NN(object):
  graphical_error_scale = 1
  pop_size = 150
  mutation_rate = 0.1#1
  crossover_rate = 0.9#2
  ni = 2
  nh = 3
  no = 1
  pat = [[[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[0, 0], [0]]]
  test_pat = [[[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[0, 0], [0]]]
  eliteN = int(pop_size * 0.15)

  def __init__(self):
    self.ai = ones(NN.ni)
    self.ah = ones(NN.nh)
    self.ao = ones(NN.no)
    self.wi = -2 + rand(NN.ni, NN.nh) * 4
    self.wo = -2 + rand(NN.nh, NN.no) * 4


  def activate(self, inputs):
    assert len(inputs) == NN.ni, 'incorrect number of inputs'
    self.ai[:] = np.tanh(inputs).copy()
    self.ah[:] = np.tanh(np.sum(self.wi.T * self.ai, axis=1))
    self.ao[:] = np.sum(self.wo.T * self.ah, axis=1) #+ 0.5
    self.ao[:] = where(self.ao > 0.5, 1.0, 0.0)
    # self.ao[:] = where(self.ao == max(self.ao), 1.0, 0.0)
    return self.ao.copy()

  def test(self, patterns):
      all_aos, targets = [], []
      for p in patterns:
          inputs = p[0]
          rounded = self.activate(inputs)
          if not (rounded-p[1]).any():
              result = '+++++'
          else: 
              result = '-----'
          print '%s %s %s %s %s %s %s' % ('Inputs:', p[0], '-->', str(self.activate(inputs)).rjust(65), 'Target', p[1], result)
          output_activations = self.activate(inputs)
          if len(output_activations) == 1:
              all_aos.append(output_activations[0])
          targets += p[1]
      return all_aos, targets

  def sumErrors(self, test_data=False):
    mse = 0.0
    all_aos = []
    if test_data:
      pat = NN.test_pat
    else:
      pat = NN.pat
    for p in pat:
      inputs = p[0]
      targets = p[1]
      output_activations = self.activate(inputs)
      all_aos.append(output_activations)
      mse += self.calcError(targets)
    # not sure of this calculation for mse
    mse = mse / len(pat) * len(pat[0])
    perc_acc = percentAcc(array(pat)[:,1], all_aos)
    outs, trgs = OutToClass(all_aos, array(pat)[:,1])
    perc_acc = 100 - percentError(outs, trgs)
    inv_err = 1.0 / mse
    return inv_err, perc_acc#, all_aos, pat

  def calcError(self, targets):
    error = 0.0
    for k in range(len(targets)):
      error += (targets[k] - self.ao[k]) ** 2
    return error

  def assignWeights(self, new_weights):
    io = 0
    np.copyto(self.wi, new_weights[io])
    io = 1
    np.copyto(self.wo, new_weights[io])

  def testWeights(self, new_weights):
    same = []
    io = 0
    for i in range(NN.ni):
      for j in range(NN.nh):
        if self.wi[i][j] != new_weights[io][i][j]:
          same.append(('I',i,j, round(self.wi[i][j],2),round(new_weights[io][i][j],2),round(self.wi[i][j] - new_weights[io][i][j],2)))
          if self.wi[i][j] > 5:
            print 'approaching infintity', i, j,  self.wi[i][j]

    io = 1
    for j in range(NN.nh):
      for k in range(NN.no):
        if self.wo[j][k] !=  new_weights[io][j][k]:
          same.append((('O',j,k), round(self.wo[j][k],2),round(new_weights[io][j][k],2),round(self.wo[j][k] - new_weights[io][j][k],2)))
          if self.wo[j][k] > 5:
            print 'approaching infintity', j, k, self.wo[j][k]
    if same:
      print same

