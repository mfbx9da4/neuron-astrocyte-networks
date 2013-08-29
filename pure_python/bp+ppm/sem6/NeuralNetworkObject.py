# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:47:21 2013

@author: david

have modified assign weights with [:] which provides in place copy rather

"""
import sys
from matplotlib.pyplot import *
from matplotlib import rc
from math import *
from numpy import arange, array
import numpy as np
import random
from random import random as r
from iris import randomdata
from random import shuffle

ni, nh, no = 4, 6, 3
MAX_ITERATIONS = 2000
N=0.02
M=0.1
M_iters = 6
Athresh, Adur = 3, 3

def sig (x):
  return tanh(x)
def dsig (y):
  return 1 - y**2

class NN:
  '''Neural network object'''
  def __init__(self, pat, testpat):
    # initialize node-activations
    self.ai = [1.0]*ni
    self.ah = [1.0]*nh
    self.ao = [1.0]*no
    self.Aah, self.Ach, self.Aao, self.Aco = [0]*nh, [0]*nh, [0]*no, [0]*no
    self.wi = [ [r() for j in range(nh)] for i in range(ni) ]
    self.wo = [ [r() for k in range(no)] for j in range(nh) ]
    self.ci = [[0.0]*nh]*ni
    self.co = [[0.0]*no]*nh
    
    self.trainpat = pat
    self.testpat = testpat
  
  def runNN (self, inputs):
    self.ai = inputs
    self.ah = [ sig(sum([ self.ai[i]*self.wi[i][j] for i in range(ni) ])) for j in range(nh)]
    self.ao = [ sig(sum([ self.ah[j]*self.wo[j][k] for j in range(nh) ])) for k in range(no)]
    return self.ao
  
  def backPropagate (self, targets):
    # calc output deltas    
    output_delta = [0.0] * no
    for k in range(no):
      error = targets[k] - self.ao[k]
      output_delta[k] =  error * dsig(self.ao[k]) 

    # calc hidden deltas
    hidden_delta = [0.0] * nh
    for j in range(nh):
      error = 0.0
      for k in range(no): 
        error += output_delta[k] * self.wo[j][k]
      hidden_delta[j] = error * dsig(self.ah[j])
      
    # update output weights
    for j in range(nh):
      for k in range(no):
        change = output_delta[k] * self.ah[j]
        self.wo[j][k] += N*change + M*self.co[j][k]
        self.co[j][k] = change
    
    # update input weights
    for i in range (ni):
      for j in range (nh):
        change = hidden_delta[j] * self.ai[i]
        self.wi[i][j] += N*change + M*self.ci[i][j]
        self.ci[i][j] = change

  def BP(self, i):
    cumerror, correct = 0.0, 0
    shuffle(self.trainpat)    
    for p in self.trainpat:
      inputs = p[0]
      targets = p[1]
      self.runNN(inputs)
      self.backPropagate(targets)
      cumerror+= self.calcerror(targets)
      correct += self.calcaccuracy(targets) 
    #if not i % 50:
      #print str(i).zfill(len(str(MAX_ITERATIONS-1))),'Combined error', cumerror
      #print str(len(pat)-correct).rjust(100),'wrong\t',float(correct)/len(pat)*100,'% correct'
    return cumerror, len(self.trainpat)-correct, float(correct)/len(self.trainpat)*100

  def NGA(self,i):
    cumerror, correct = 0.0, 0
    shuffle(self.trainpat)
    for p in self.trainpat:
      inputs = p[0]
      targets = p[1]
      self.astrocyteactions(inputs, targets)
      cumerror+= self.calcerror(targets)
      correct += self.calcaccuracy(targets)
    return cumerror, len(self.trainpat)-correct, float(correct)/len(self.trainpat)*100

  def astrocyteactions (self, inputs, targets):
    for m in range(M_iters):
      self.ai = inputs
      for j in range(nh):
        self.ah[j] = sig(sum([ self.ai[i]*self.wi[i][j] for i in range(ni) ]))
        if self.ah[j] > 0: self.Aah[j] +=1
        else: self.Aah[j] -=1
        if self.Aah[j] == Athresh:
          self.Ach[j] = Adur
        elif self.Aah[j] == -Athresh:
          self.Ach[j] = -Adur  
        if self.Ach[j] > 0:
          for i in range(ni):
            self.wi[i][j] += self.wi[i][j]*0.25
          self.Ach[j] -=1
        elif self.Ach[j] < 0:
          for i in range(ni):
            self.wi[i][j] += self.wi[i][j]*-0.5
          self.Ach[j] +=1
         
      for k in range(no):
        self.ao[k] = sig(sum([ self.ah[j]*self.wo[j][k] for j in range(nh) ]))
        if self.ao[k] > 0: self.Aao[k] +=1
        else: self.Aao[k] -=1
        if self.Aao[k] == Athresh:
          self.Aco[k] = Adur
        elif self.Aao[k] == -Athresh:
          self.Aco[k] = -Adur
        if self.Aco[k] > 0:
          for j in range(nh):
            self.wo[j][k] += self.wo[j][k]*0.25
          self.Aco[k] -=1
        elif self.Aco[k] < 0:
          for j in range(nh):
            self.wo[j][k] += self.wo[j][k]*-0.5
          self.Aco[k] +=1
      if m == M_iters-1:
        error = 0.0
        for k in range(len(targets)):
          error += 0.5 * (targets[k]-self.ao[k])**2
    return error
  
  def test(self):
    cumerror, correct = 0.0, 0
    shuffle(self.testpat)    
    for p in self.testpat:
      targets = p[1]
      #print 'Inputs:', p[0], '-->', str(self.runNN(inputs)).rjust(60), '\tTarget', targets
      cumerror+= self.calcerror(targets)
      correct += self.calcaccuracy(targets)
    #print 'Combined error', cumerror
    #print str(len(testpat)-correct).rjust(100),'wrong\t',float(correct)/len(testpat)*100,'% correct'
    return [cumerror, len(self.testpat)-correct, float(correct)/len(self.testpat)*100]
  
  def calcerror(self, targets):
    error = 0.0
    for k in range(len(targets)):
      error += 0.5 * (targets[k]-self.ao[k])**2
    return error

  def calcaccuracy(self, targets):
    iscorrect = 1
    for k in range(len(targets)):
      if targets[k]-round(self.ao[k]):
        iscorrect = 0
    return iscorrect
  
  def train (self, isNGA):
    bperrs, ngaerrs, bpwngs, ngawngs, bpaccs, ngaaccs = [],[],[],[],[],[]
    if isNGA:
      for i in range(MAX_ITERATIONS):
        err1, wng1, acc1 = self.BP(i)
        bperrs += [err1]; bpwngs += [wng1]; bpaccs += [acc1]
        err2, wng2, acc2 = self.NGA(i)
        ngaerrs += [err2]; ngawngs += [wng2]; ngaaccs += [acc2]
      return [bperrs, ngaerrs], [bpwngs, ngawngs], [bpaccs, ngaaccs]
    else:
      for i in range(MAX_ITERATIONS):
        err1, wng1, acc1 = self.BP(i)
        bperrs += [err1]; bpwngs += [wng1]; bpaccs += [acc1]
      blank_values = [0]*MAX_ITERATIONS # to satisfy later operations
      return [bperrs, blank_values], [bpwngs, blank_values], [bpaccs, blank_values]
