from matplotlib.pyplot import *
from math import *
import random
from random import random as r
from training_patterns import boolpat as pat
#pat = [
   #[[0,0], [0]],
   #[[0,1], [1]],
   #[[1,0], [1]],
   #[[1,1], [0]]
 #]

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
      
  
  def backPropagate (self, targets, N, M):
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
        
    # calc combined error
    # 1/2 for differential convenience & **2 for modulus
    error = 0.0
    for k in range(len(targets)):
      error += 0.5 * (targets[k]-self.ao[k])**2
    # accuracy percentage
    iscorrect = 1
    for k in range(len(targets)):
      if targets[k]-round(self.ao[k]):
        iscorrect = 0
    return error, iscorrect

  def BP(self, patterns, N, M,i):
    cumerror, correct = 0.0, 0
    for p in patterns:
      inputs = p[0]
      targets = p[1]
      self.runNN(inputs)
      instance_error, instance_iscorrect = self.backPropagate(targets, N, M)
      cumerror+= instance_error
      correct += instance_iscorrect
    if i % 50 == 0:
      print str(i).zfill(len(str(max_iterations-1))),'Combined error', cumerror
      print '\t\t\t\t\t\t\t\t',len(patterns)-correct,'wrong\t',float(correct)/len(patterns)*100,'% correct'
    return cumerror, correct

  def NGA(self, pat,i ):
    error = 0.0
    for p in pat:
      inputs = p[0]
      targets = p[1]
      error += self.astrocyteactions(inputs, targets)
    return error

  def astrocyteactions (self, inputs, targets):
    for m in range(M_iters):
      self.ai = inputs
      for j in range(nh):
        self.ah[j] = sig(sum([ self.ai[i]*self.wi[i][j] for i in range(ni) ]))
        if self.ah[j] > 0: self.Aah[j] +=1
        else: self.Aah[j] -=1
        if self.Aah[j] == Athresh:
          self.Ach[j] = Adur
          self.Aah[j] = 0
        elif self.Aah[j] == -Athresh:
          self.Ach[j] = -Adur
          self.Aah[j] = 0
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
          self.Aao[k] = 0
        elif self.Aao[k] == -Athresh:
          self.Aco[k] = -Adur
          self.Aao[k] = 0
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
    print
    print 'Output weights:'
    for j in range(nh):
      print map(lambda x: round(x,2),self.wo[j])
    print ''
  
  def test(self, patterns):
    for p in patterns:
      inputs = p[0]
      print 'Inputs:', p[0], '-->', str(self.runNN(inputs)).rjust(60), '\tTarget', p[1]
  
  def train (self, patterns, trials):
    figure()
    for T in range(1,trials):
      errs, bperrs, ngaerrs, corrects = [],[],[],[]
      for i in range(max_iterations):
        bperr, correct = self.BP(patterns, N, M,i)
        bperrs += [bperr]
        corrects += [correct]
        if T % 2: ngaerrs += [self.NGA(patterns,i)]
      self.test(patterns)
      self.weights()
      g = subplot(int(sqrt(trials)),2,T)
      plot(bperrs, label='Backpropagation')
      plot(ngaerrs, label='NGA')
      if T%2: legend(loc='upper right')
      ylabel('Error')
      xlabel('Iterations')
      #g.set_ylim(0,1)
      self.__init__()
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
ni, nh, no = len(pat[0][0]),len(pat[0][1])*2,len(pat[0][1])
trials = 2+1

def main ():
  myNN = NN()
  myNN.train(pat, trials)
  
  

if __name__ == "__main__":
    main()
