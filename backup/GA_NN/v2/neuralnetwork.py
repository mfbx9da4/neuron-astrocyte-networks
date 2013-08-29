import random

from pylab import array, ones, zeros, where
import numpy as np

from utilities import sigmoid, randomizeMatrix
from iris import pat as train_pat


class NN(object):
  graphical_error_scale = 100
  pop_size = 150
  mutation_rate = 0.1
  crossover_rate = 0.9
  ni = 4
  nh = 6
  no = 1
  pat = train_pat
  eliteN = int(pop_size * 0.15)

  def __init__(self):
    self.ai = ones(NN.ni)
    self.ah = ones(NN.nh)
    self.ao = ones(NN.no)
    self.wi = zeros((NN.ni, NN.nh))
    self.wo = zeros((NN.nh, NN.no))
    randomizeMatrix(self.wi, -0.2, 0.2)
    randomizeMatrix(self.wo, -2.0, 2.0)


  def runNN(self, inputs):
    assert len(inputs) == NN.ni, 'incorrect number of inputs'
    self.ai = np.tanh(inputs)
    for j in range(NN.nh):
        self.ah[j] = sigmoid(sum([self.ai[i] * self.wi[i][j] for i in range(NN.ni)]))
    for k in range(NN.no):
        self.ao[k] = sigmoid(sum([self.ah[j] * self.wo[j][k] for j in range(NN.nh)]))
    return self.ao

  def test(self, patterns):
      results, targets = [], []
      for p in patterns:
          inputs = p[0]
          rounded = [round(i) for i in self.runNN(inputs)]
          if rounded == p[1]: 
              result = '+++++'
          else: 
              result = '-----'
          print '%s %s %s %s %s %s %s' % ('Inputs:', p[0], '-->', str(self.runNN(inputs)).rjust(65), 'Target', p[1], result)
          output_activations = self.runNN(inputs)
          if len(output_activations) == 1:
              results.append(output_activations[0])
          targets += p[1]
      return results, targets

  def sumErrors(self):
    mse = 0.0
    for p in NN.pat:
      inputs = p[0]
      targets = p[1]
      self.runNN(inputs)
      mse += self.calcMse(targets)
    inverr = 1.0 / mse
    return inverr

  def calcMse(self, targets):
    error = 0.0
    for k in range(len(targets)):
      error += 0.5 * (targets[k] - self.ao[k]) ** 2
    return error

  def assignWeights(self, new_weights):
    # can replace this with just wi = new.copy()
    io = 0
    for i in range(NN.ni):
      for j in range(NN.nh):
        self.wi[i][j] = new_weights[io][i][j]
    io = 1
    for j in range(NN.nh):
      for k in range(NN.no):
        self.wo[j][k] = new_weights[io][j][k]

  def testWeights(self, new_weights):
    same = []
    io = 0
    for i in range(NN.ni):
      for j in range(NN.nh):
        if self.wi[i][j] != new_weights[io][i][j]:
          same.append(('I',i,j, round(self.wi[i][j],2),round(new_weights[io][i][j],2),round(self.wi[i][j] - new_weights[io][i][j],2)))

    io = 1
    for j in range(NN.nh):
      for k in range(NN.no):
        if self.wo[j][k] !=  new_weights[io][j][k]:
          same.append((('O',j,k), round(self.wo[j][k],2),round(new_weights[io][j][k],2),round(self.wo[j][k] - new_weights[io][j][k],2)))
    if same:
      print same

  def printWeights(self):
    print 'Input weights:'
    for i in range(NN.ni):
      print self.wi[i]
    print
    print 'Output weights:'
    for j in range(NN.nh):
      print self.wo[j]
    print ''
