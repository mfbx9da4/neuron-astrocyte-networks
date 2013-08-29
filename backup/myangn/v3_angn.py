from matplotlib.pyplot import *
from math import *
from numpy import arange, array
import numpy as np
import random
from random import random as r
from training_patterns import boolpat as pat
from training_patterns import testpat

class NN:
  def __init__(self):
    # initialize node-activations
    self.ai = [1.0]*ni
    self.ah = [1.0]*nh
    self.ao = [1.0]*no
    self.Aah, self.Ach, self.Aao, self.Aco = [0]*nh, [0]*nh, [0]*no, [0]*no
    self.wi = [ [r() for j in range(nh)] for i in range(ni) ]
    self.wo = [ [r() for k in range(no)] for j in range(nh) ]
    self.ci = [[0.0]*nh]*ni
    self.co = [[0.0]*no]*nh
  
  def runNN (self, inputs):
    self.ai = inputs
    self.ah = [ sig(sum([ self.ai[i]*self.wi[i][j] for i in range(ni) ])) for j in range(nh)]
    self.ao = [ sig(sum([ self.ah[j]*self.wo[j][k] for j in range(nh) ])) for k in range(no)]
    return self.ao
  
  def backPropagate (self, targets):
    output_deltas = [0.0] * no
    for k in range(no):
      error = targets[k] - self.ao[k]
      output_deltas[k] =  error * dsig(self.ao[k]) 

    for j in range(nh):
      for k in range(no):
        change = output_deltas[k] * self.ah[j]
        self.wo[j][k] += N*change + M*self.co[j][k]
        self.co[j][k] = change

    # calc hidden deltas
    hidden_deltas = [0.0] * nh
    for j in range(nh):
      for k in range(no):
        error = output_deltas[k] * self.wo[j][k]
      hidden_deltas[j] = error * dsig(self.ah[j])
    
    #update input weights
    for i in range (ni):
      for j in range (nh):
        change = hidden_deltas[j] * self.ai[i]
        self.wi[i][j] += N*change + M*self.ci[i][j]
        self.ci[i][j] = change

  def BP(self, i):
    cumerror, correct = 0.0, 0
    for p in pat:
      inputs = p[0]
      targets = p[1]
      self.runNN(inputs)
      self.backPropagate(targets)
      cumerror+= self.calcerror(targets)
      correct += self.calcaccuracy(targets)
    if not i % 50:
      print str(i).zfill(len(str(max_iterations-1))),'Combined error', cumerror
      print str(len(pat)-correct).rjust(100),'wrong\t',float(correct)/len(pat)*100,'% correct'
    return cumerror, len(pat)-correct, float(correct)/len(pat)*100

  def NGA(self,i):
    cumerror, correct = 0.0, 0
    for p in pat:
      inputs = p[0]
      targets = p[1]
      self.astrocyteactions(inputs, targets)
      cumerror+= self.calcerror(targets)
      correct += self.calcaccuracy(targets)
    return cumerror, len(pat)-correct, float(correct)/len(pat)*100

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
      
  def weights(self):
    print 'Input weights:'
    for i in range(ni):
      print map(lambda x: round(x,2),self.wi[i])
    print '\nOutput weights:'
    for j in range(nh):
      print map(lambda x: round(x,2),self.wo[j])
    print 
  
  def test(self):
    cumerror, correct = 0.0, 0
    for p in testpat:
      inputs = p[0]
      targets = p[1]
      print 'Inputs:', p[0], '-->', str(self.runNN(inputs)).rjust(60), '\tTarget', targets
      cumerror+= self.calcerror(targets)
      correct += self.calcaccuracy(targets)
    print 'Combined error', cumerror
    print str(len(testpat)-correct).rjust(100),'wrong\t',float(correct)/len(testpat)*100,'% correct'
    return [cumerror, len(testpat)-correct, float(correct)/len(testpat)*100]
  
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
    errs, bperrs, ngaerrs, bpwngs, ngawngs, bpaccs, ngaaccs = [],[],[],[],[],[],[]
    for i in range(max_iterations):
      err1, wng1, acc1 = self.BP(i)
      bperrs += [err1]; bpwngs += [wng1]; bpaccs += [acc1]
      if isNGA: 
        err2, wng2, acc2 = self.NGA(i)
        ngaerrs += [err2]; ngawngs += [wng2]; ngaaccs += [acc2]
    if isNGA: return [bperrs, ngaerrs], [bpwngs, ngawngs], [bpaccs, ngaaccs]
    else: return [bperrs], [bpwngs], [bpaccs]
      
      
def plotalltrials(ngatrainerrs, bptrainerrs):
  figure()
  suptitle('Training error', fontsize=18)
  i = 1
  for trial in ngatrainerrs:
    subplot(int(ceil(trials/2)),2,i)
    plot(trial[0], label='Backpropagation')
    i+=2
  i = 2
  for trial in ngatrainerrs:
    g = subplot(int(ceil(trials/2)),2,i)
    l1 = plot(trial[0], label='BP')
    l2 = plot(trial[1], label='NGA')
    i+=2
  g.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
  
  show()
  
def plottrainavgs(ngas,bps,Title='',ylimit=0):
  ngas = array(ngas)
  bps = array(bps)
  ngaavgs = array([ np.mean(ngas[:,0,:], axis=0), np.mean(ngas[:,1,:], axis=0) ])
  ngastds = array([ np.std(ngas[:,0,:], axis=0), np.std(ngas[:,1,:], axis=0) ])/sqrt(max_iterations)
  bpavgs = np.mean(bps[:,0,:], axis=0)
  bpstds = np.std(bps[:,0,:], axis=0)/sqrt(max_iterations)
  figure()
  suptitle(Title, fontsize=18)
  subplot(1,2,1); title('BP only')
  if ylimit: ylim((0,100))
  errorbar(range(len(bpavgs)),bpavgs, yerr=bpstds, fmt='b-')
  subplot(1,2,2); title('NGA & BP')
  if ylimit: ylim((0,100))
  errorbar(range(len(ngaavgs[0])),ngaavgs[0], yerr=ngastds[0], fmt='b-')
  errorbar(range(len(ngaavgs[1])),ngaavgs[1], yerr=ngastds[1], fmt='g-')
  
def sig (x):
  return tanh(x)
def dsig (y):
  return 1 - y**2

pat = [[[0,0], [0]],[[0,1], [1]],[[1,0], [1]],[[1,1], [0]]];testpat = [[[0,0], [0]],[[0,1], [1]],[[1,0], [1]],[[1,1], [0]]]
max_iterations = 5000
N=0.2
M=0.1
M_iters = 6
Athresh, Adur = 3, 3
ni, nh, no = len(pat[0][0]),len(pat[0][1])*2,len(pat[0][1])
trials = 100

def main ():
  ngatrainerrs, ngatrainwngs, bptrainaccs, ngatrainaccs, bptrainerrs, bptrainwngs = [],[],[],[],[],[]
  ngatesterrs, ngatestwngs, bptestaccs, ngatestaccs, bptesterrs, bptestwngs = [],[],[],[],[],[]
  isNGA = True
  for T in range(trials):
    myNN = NN()
    if isNGA: 
      err1, wng1, acc1 = myNN.train(isNGA)
      ngatrainerrs.append(err1)
      ngatrainwngs.append(wng1)
      ngatrainaccs.append(acc1)
      err2, wng2, acc2 = myNN.test()
      ngatesterrs.append(err2)
      ngatestwngs.append(wng2)
      ngatestaccs.append(acc2)
    else: 
      err3, wng3, acc3 = myNN.train(isNGA)
      bptrainerrs.append(err3)
      bptrainwngs.append(wng3)
      bptrainaccs.append(acc3)
      err4, wng4, acc4 = myNN.test()
      bptesterrs.append(err4)
      bptestwngs.append(wng4)
      bptestaccs.append(acc4)
    #myNN.weights()
    isNGA = not isNGA
  
  #plotalltrials(ngatrainerrs, bptrainerrs)
  plottrainavgs(ngatrainerrs, bptrainerrs, Title='Training error')
  plottrainavgs(ngatrainwngs, bptrainwngs, Title='Training wrongs (instances wrong) vs iterations')
  plottrainavgs(ngatrainaccs, bptrainaccs, Title='Training accuracies (%) vs iterations', ylimit=1)
  figure(); plot(ngatesterrs); plot(bptesterrs)
  figure(); plot(ngatestwngs); plot(bptestwngs)
  figure(); plot(ngatestaccs); plot(bptestaccs)
  #plottestavgs(ngatesterrs, bptesterrs, Title='Test error')
  #plottestavgs(ngatestwngs, bptestwngs, Title='Test wrongs (instances wrong) vs iterations')
  #plottestavgs(ngatestaccs, bptestaccs, Title='Test accuracies (%) vs iterations', ylimit=1)
  show()



      
  
if __name__ == "__main__":
    main()
