from matplotlib.pyplot import *
from math import *
from numpy import arange, array
import numpy as np
import random
from random import random as r
from training_patterns import boolpat as pat
from training_patterns import testpat
#pat = [[[0,0], [0]],[[0,1], [1]],[[1,0], [1]],[[1,1], [0]]]
#testpat = [[[0,0], [0]],[[0,1], [1]],[[1,0], [1]],[[1,1], [0]]]

class NN:
  def __init__(self):
    # initialize node-activations
    self.ai = [1.0]*ni
    self.ah = [1.0]*nh
    self.ao = [1.0]*no
    self.Aah, self.Ach, self.Aao, self.Aco = [0]*nh, [0]*nh, [0]*no, [0]*no
    self.AAah, self.AAch = [ [0]*(nh-(j+1)) for j in range(nh-1) ],[ [0]*(nh-(j+1)) for j in range(nh-1) ]
    self.AAao, self.AAco = [ [0]*(no-(k+1)) for k in range(no-1) ],[ [0]*(no-(k+1)) for k in range(no-1) ]
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
    return cumerror, correct

  def NGA(self,i):
    cumerror, correct = 0.0, 0
    for p in pat:
      inputs = p[0]
      targets = p[1]
      self.astrocyteactions(inputs, targets)
      cumerror+= self.calcerror(targets)
      correct += self.calcaccuracy(targets)
    return cumerror, correct

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
        
        # For each AA
        if not j == nh-1:
          for j2 in range(len(self.AAah[j])):
            #check Ac(j) & Ac(j2): ++/-- AAa
            if self.Ach[j]>0 and self.Ach[j2]>0 and self.Ach[j] == self.Ach[j2]:
              self.AAah[j][j2] += 1
            elif self.Ach[j]<0 and self.Ach[j2]<0 and self.Ach[j] == self.Ach[j2]:
              self.AAah[j][j2] -= 1
            #check AAa(j,j2): AAdur/-AAdur AAc
            if self.AAah[j][j2] == AAthresh: self.AAch[j][j2] = AAdur
            elif self.AAah[j][j2] == -AAthresh: self.AAch[j][j2] = -AAdur
            # check AAc(j,j2): check Ac(j) & Ac(j2): Adur/-Adur ( Ac(j) & Ac(j2) )
            if self.AAch[j][j2] == AAdur:
              if self.Aah[j]==Athresh or self.Aah[j2]==Athresh:
                self.Aah[j]=Athresh; self.Aah[j2]=Athresh
              self.AAch[j][j2] -=1
            elif self.AAch[j][j2] == -AAdur:
              if self.Aah[j]==-Athresh or self.Aah[j2]==-Athresh:
                self.Aah[j]=-Athresh; self.Aah[j2]=-Athresh
              self.AAch[j][j2] +=1
         
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
        
        # For each AA
        if not k==no-1:
          for k2 in range(len(self.AAao[k])):
            #check Ac(k) & Ac(k2): ++/-- AAa
            if self.Aco[k]>0 and self.Aco[k2]>0 and self.Aco[k] == self.Aco[k2]:
              self.AAao[k][k2] += 1
            elif self.Aco[k]<0 and self.Aco[k2]<0 and self.Aco[k] == self.Aco[k2]:
              self.AAao[k][k2] -= 1
            #check AAa(k,k2): AAdur/-AAdur AAc
            if self.AAao[k][k2] == AAthresh: self.AAco[k][k2] = AAdur
            elif self.AAao[k][k2] == -AAthresh: self.AAco[k][k2] = -AAdur
            # check AAc(k,k2): check Ac(k) & Ac(k2): Adur/-Adur ( Ac(k) & Ac(k2) )
            if self.AAco[k][k2] == AAdur:
              if self.Aao[k]==Athresh or self.Aao[k2]==Athresh:
                self.Aao[k]=Athresh; self.Aao[k2]=Athresh
              self.AAco[k][k2] -=1
            elif self.AAco[k][k2] == -AAdur:
              if self.Aao[k]==-Athresh or self.Aao[k2]==-Athresh:
                self.Aao[k]=-Athresh; self.Aao[k2]=-Athresh
              self.AAco[k][k2] +=1
      
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
    return cumerror, correct
  
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
    errs, bperrs, ngaerrs, bpcorrects, ngacorrects = [],[],[],[],[]
    for i in range(max_iterations):
      err, correct = self.BP(i)
      bperrs += [err]; bpcorrects += [correct]
      if isNGA: 
        err, correct = self.NGA(i)
        ngaerrs += [err]; ngacorrects += [correct]
    return bperrs, ngaerrs
      
      
def plotresults(trainerrs, testerrs):
    isNGA = True
    figure()
    for trial in range(trials):
      title('Training error')
      g = subplot(int(ceil(trials/2)),2,trial+1)
      plot(trainerrs[trial][0], label='Backpropagation')
      if isNGA: plot(trainerrs[trial][1], label='NGA'); legend(loc='right')
      xlabel('Iterations')
      isNGA = not isNGA
      #g.set_ylim(0,1)
    #for trial in range(1,trials):
      #figure()
      #title('Test error')
      #x = arange(1)
      #bar(x,testerrs[trial][0])
      #xticks(x+0.5,['Test'])
      #ylabel('Error')
    show()

def sig (x):
  return tanh(x)
def dsig (y):
  return 1 - y**2

max_iterations = 3000
N=0.2
M=0.1
M_iters = 6
Athresh, Adur = 3, 3
AAthresh, AAdur = Athresh, Adur
ni, nh, no = len(pat[0][0]),len(pat[0][1])*2,len(pat[0][1])
trials = 2

def main ():
  trainerrs, testerrs = [], []
  isNGA = True
  for T in range(trials):
    myNN = NN()
    trainerrs += [myNN.train(isNGA)]
    testerrs += [myNN.test()]
    myNN.weights()
    isNGA = not isNGA
  plotresults(trainerrs, testerrs)
  

if __name__ == "__main__":
    main()
