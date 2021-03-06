from operator import itemgetter, attrgetter
import math
from math import copysign
from random import *
import GADANN_test as tst
from training_patterns import boolpat as pat
from training_patterns import boolpat as testpat
import timeit
from timeit import Timer as t
from matplotlib.pyplot import *
from numpy import array, arange, ones, random

def sigmoid (x):
  return math.tanh(x)

class NN:
  # ni,nh,no = n of input (i), hidden (h) and output (o) nodes
  # ai,ah,ao = output value for nodes in i,h,o layers
  # Ah, Ao = Astrocyte activation value/counter of partner neuronal activity (h,o layers)
  # Ah_countdown = counts down duration of astrocyte action in itererations
  # wi,wo = input and output weights initialized to random values
  def __init__(self, NI, NH, NO):
    self.ni = NI
    self.nh = NH
    self.no = NO
    self.ai = ones(ni)
    self.ah = ones(nh) # a.cumsum( acis = 1) cumsum of rows
    self.ao = ones(no) # a.min( axis = 0) min in cols
    self.Ah = [0]*self.nh # self.Ah[j].sum() to sum
    self.Ao = [0]*self.no # for element in a.flat # gives all elements # a.ravel()/a.shape = 
    self.Ah_countdwn = [0]*self.nh
    self.Ao_countdwn = [0]*self.no
    self.wi = random.uniform(-2,2,(ni,nh))
    self.wo = random.uniform(-2,2,(nh,no))

  def runNN (self, inputs):
    if len(inputs) != self.ni:
      print 'incorrect number of inputs'
    for i in range(self.ni):
      self.ai[i] = inputs[i]
    for j in range(self.nh):
      self.ah[j] = sigmoid(sum([ self.ai[i]*self.wi[i][j] for i in range(self.ni) ]))     
    for k in range(self.no):
      self.ao[k] = sigmoid(sum([ self.ah[j]*self.wo[j][k] for j in range(self.nh) ]))
    return self.ao
    
  def runNGA (self, inputs):
    if len(inputs) != self.ni:
      print 'incorrect number of inputs'
    for i in range(self.ni):
      self.ai[i] = inputs[i]
      
    for j in range(self.nh):
      self.ah[j] = sigmoid(sum([ self.ai[i] * self.wi[i][j] for i in range(self.ni) ]))
      # check neuron activation
      if self.ah[j] > 0: self.Ah[j] +=1
      else: self.Ah[j] -=1

      # check if astrocyte should be activated
      if self.Ah[j] >= Athresh:
        self.Ah_countdwn[j] = Adur
        self.Ah[j] = 0
      elif self.Ah[j] <= -Athresh:
        self.Ah_countdwn[j] = -Adur
        self.Ah[j] = 0
        
      # check if astrocyte is active --> perform actions
      if self.Ah_countdwn[j] > 0:
        for i in range(ni):
          if not self.wi[i][j] > wl:
            self.wi[i][j] += (wl-self.wi[i][j])*0.25
        self.Ah_countdwn[j] -= 1
      elif self.Ah_countdwn[j] < 0:
        for i in range(ni):
          if not self.wi[i][j] < -wl:
            self.wi[i][j] -= (wl+self.wi[i][j])*0.5
        self.Ah_countdwn[j] += 1
        
    for k in range(self.no):
      self.ao[k] = sigmoid(sum([ self.ah[j] * self.wo[j][k] for j in range(self.nh) ]))
      # check neuron activation
      if self.ao[k] > 0: self.Ao[k] += 1
      else: self.Ao[k] -= 1

      # check if astrocyte should be activated
      if self.Ao[k] >= Athresh: 
        self.Ao_countdwn[k] = Adur
        self.Ao[k] = 0
      elif self.Ao[k] <= -Athresh: 
        self.Ao_countdwn[k] = -Adur
        self.Ao[k] = 0
      
      # check if astrocyte is active --> perform actions
      if self.Ao_countdwn[k] > 0:
        for j in range(nh):
          if not self.wo[j][k] > wl:
            self.wo[j][k] += (wl-self.wo[j][k])*0.25
        self.Ao_countdwn[k] -= 1
      elif self.Ao_countdwn[k] < 0:
        for j in range(nh):
          if not self.wo[j][k] < -wl:
            self.wo[j][k] -= (wl+self.wo[j][k])*0.5
        self.Ao_countdwn[k] += 1

  def test(self, patterns):
    results, targets = [], []
    for p in patterns:
      inputs = p[0]
      rounded = [ round(i) for i in self.runNN(inputs) ]
      if rounded == p[1]: result = '+++++'
      else: result = '-----'
      print '%s %s %s %s %s %s %s' %( 'Inputs:', p[0], '-->', str([ round(i,3) for i in self.runNN(inputs) ]).rjust(65), 'Target', p[1], result)
      results+= eval(repr([self.runNN(inputs)]))
      targets += [p[1]]
    return results, targets

  def sumErrors (self):
    error = 0.0
    for p in pat:
      inputs = p[0]
      targets = p[1]
      self.runNN(inputs)
      error += self.calcError(targets)
    inverr = 1.0/error
    return inverr

  def calcError (self, targets):
    error = 0.0
    for k in range(len(targets)):
      error += 0.5 * (targets[k]-self.ao[k])**2
    return error

  def assignWeights (self, weights, I):
    io = 0
    for i in range(self.ni):
      for j in range(self.nh):
        self.wi[i][j] = weights[I][io][i][j]
    io = 1
    for j in range(self.nh):
      for k in range(self.no):
        self.wo[j][k] = weights[I][io][j][k]

  def testWeights (self, weights, I):
    same = []
    io = 0
    for i in range(self.ni):
      for j in range(self.nh):
        if self.wi[i][j] != weights[I][io][i][j]:
          same.append(('I',i,j, round(self.wi[i][j],2),round(weights[I][io][i][j],2),round(self.wi[i][j] - weights[I][io][i][j],2)))
    io = 1
    for j in range(self.nh):
      for k in range(self.no):
        if self.wo[j][k] !=  weights[I][io][j][k]:
          same.append((('O',j,k), round(self.wo[j][k],2),round(weights[I][io][j][k],2),round(self.wo[j][k] - weights[I][io][j][k],2)))
    if same != []:
      print I,'\n',same

def roulette (fitnessScores):
  cumalativeFitness = 0.0
  r = random.random()
  for i in range(len(fitnessScores)): 
    cumalativeFitness += fitnessScores[i]
    if cumalativeFitness > r: 
      return i
      
def calcFit (numbers):  # each fitness is a fraction of the total error
  total, fitnesses = sum(numbers), []
  for i in range(len(numbers)):           
    fitnesses.append(numbers[i]/total)
  return fitnesses

# Takes a population of NN objects, returns a list of the weights tupled with their 
# fitness scores.
# Calculates the MSE for each instance by subtraction from target value and sums over all
# instances. 
# Fitness score for each individual is a fraction of the total error of the population.
def pairPop (pop):
  weights, errors = [], []
  for i in range(len(pop)):                 
    weights.append([pop[i].wi,pop[i].wo])   
    errors.append(pop[i].sumErrors())       
  fitnesses = calcFit(errors)               
  for i in range(int(pop_size*0.15)): 
    print str(i).zfill(2), '1/sum(MSEs)', str(errors[i]).rjust(15), str(int(errors[i]*graphical_error_scale)*'-').rjust(20), 'fitness'.rjust(12), str(fitnesses[i]).rjust(17), str(int(fitnesses[i]*1000)*'-').rjust(20)
  print 
  del pop
  return zip(weights, errors, fitnesses)
  
# Takes weights for a pop and ranks the weights 
# Returns the weights paired and sorted in order of fitness
def rankPop (newpopW):
  print 'RANKPOP START'
  pop, errors, copy = [ NN(ni,nh,no) for i in range(pop_size) ], [], []
  for i in range(pop_size): copy.append(newpopW[i])
  for i in range(pop_size):  
    pop[i].assignWeights(newpopW, i)                                    # each individual is assigned the weights generated from previous iteration
    pop[i].testWeights(newpopW, i)
  for i in range(pop_size):  
    pop[i].testWeights(newpopW, i)
  tst.newpopWchanged(newpopW, copy) 
  pairedPop = pairPop(pop)                                              # the fitness of these weights is calculated and tupled with the weights
  rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True)   # weights are sorted in descending order of fitness (fittest first)
  errors = [ eval(repr(x[1])) for x in rankedPop ]
  print 'END RANKPOP'
  return rankedPop, eval(repr(rankedPop[0][1])), float(sum(errors))/float(len(errors))

def iteratePop (rankedPop):
  rankedWeights = [ item[0] for item in rankedPop]
  fitnessScores = [ item[-1] for item in rankedPop]
  newpopW = [ eval(repr(x)) for x in rankedWeights[:int(pop_size*0.15)] ]
  while len(newpopW) <= pop_size:                                       # Breed two randomly selected but different chromos until pop_size reached
    ch1, ch2 = [], []
    index1 = roulette(fitnessScores)                                    
    index2 = roulette(fitnessScores)
    while index1 == index2:                                             # ensures different chromos are used for breeeding 
      index2 = roulette(fitnessScores)
    #index1, index2 = 3,4
    ch1.extend(eval(repr(rankedWeights[index1])))
    ch2.extend(eval(repr(rankedWeights[index2])))
    if random.random() < crossover_rate: 
      ch1, ch2 = crossover(ch1, ch2)
    mutate(ch1)  
    mutate(ch2)  
    newpopW.append(ch1)
    newpopW.append(ch2)
  tst.dezip(rankedPop, rankedWeights, fitnessScores)
  #tst.elitism(rankedWeights, newpopW)
  return newpopW

def NGA(newpopW):
  print 'NGA START'
  pop = [ NN(ni,nh,no) for i in range(pop_size) ]
  #pop = [ NN(ni,nh,no) ]*pop_size
  weights = []
  for i in range(pop_size):
    pop[i].assignWeights(newpopW, i)
    pop[i].testWeights(newpopW, i)
    for p in range(len(pat)):
      inputs, targets = pat[p][0], pat[p][1]
      for m in range(m_iters):
        pop[i].runNGA(inputs)
    weights.append([pop[i].wi,pop[i].wo])
  del pop
  print 'END NGA'
  return weights

graphical_error_scale = 500
max_iterations = 1000
pop_size = 100
mutation_rate = 0.1
crossover_rate = 0.8
ni, nh, no = 4,6,3 # number of input (i), hidden (h) and output (o) nodes
m_iters = 6 # number of iterations per instance
Athresh = 3 # threshold count for activation of astrocyte
Adur = 2 # duration of action of astrocyte
wl = 2.0 # weight limit

def main ():
  
  # Generates a population of NNs with random weights and rank them based on their error
  pop = [ NN(ni,nh,no) for i in range(pop_size) ] 
  pairedPop = pairPop(pop)
  rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True)
  tst.correctlyranked(rankedPop)
  
  # Keep iterating new pops until max_iterations
  iters = 0
  tops, avgs = [], []
  newpopW = iteratePop(rankedPop)
  
  while iters != max_iterations:
  
    if iters%1 == 0:
      print 'Iteration'.rjust(150), iters

    newpopW = NGA(newpopW)
    rankedPop, toperr, avgerr = rankPop(newpopW)
    newpopW = iteratePop(rankedPop)

    tops.append(toperr)
    avgs.append(avgerr)
    iters+=1
  
  # test a NN with the fittest weights
  tester = NN (ni,nh,no)
  fittestWeights = [ x[0] for x in rankedPop ]
  tester.assignWeights(fittestWeights, 0)
  results, targets = tester.test(testpat)
  plotresults(results,targets)
  
def plotresults(results,targets):
  for i in range(len(results)): 
    print results[i], targets[i]
  lp = len(pat)
  x = arange(0,lp/3)
  title2 = 'Test after '+str(iters)+' iterations'
  title(title2)
  for i in range(3):
    sp = '3'+'1'+str(i)
    subplot(int(sp))
    ylabel('Node output')
    xlabel('Instances')
    for x in range(lp/3):
      plot( results[x+(i*(lp/3))], label='Instance'+str(x))
      print results[x+(i*(lp/3))]
    plot( targets[i*(lp/3)], 'kx-', markersize = 10, linewidth = 3)
  legend(loc = 'upper right')
  annotate(s='Target Values', xy = (110, 0),color = 'black', family = 'sans-serif', size  ='small')
  annotate(s='Test Values', xy = (110, 0.5),color = 'red', family = 'sans-serif', size  ='small', weight = 'bold')
  figure(2)
  title('Top individual error evolution')
  title('Population average error evolution')
  plot( avgs, '-g', linewidth = 0.5)
  plot( tops, '-r', linewidth = 2)
  ylabel('Inverse error')
  xlabel('Iterations')
  show()
  
  print 'max_iterations',max_iterations,'\tpop_size',pop_size,'pop_size*0.15',int(pop_size*0.15),'\tmutation_rate',mutation_rate,'crossover_rate',crossover_rate,'ni, nh, no',ni, nh, no
  

def crossover (m1, m2):
  r = random.randint(0, (ni*nh)+(nh*no) ) # ni*nh+nh*no = total n of weights
  output1 = [ [[0.0]*nh]*ni ,[[0.0]*no]*nh ]
  output2 = [ [[0.0]*nh]*ni ,[[0.0]*no]*nh ]
  for i in range(len(m1)):
    for j in range(len(m1[i])):
      for k in range(len(m1[i][j])):
        if r >= 0:
          output1[i][j][k] = m1[i][j][k]
          output2[i][j][k] = m2[i][j][k]
        elif r < 0:
          output1[i][j][k] = m2[i][j][k]
          output2[i][j][k] = m1[i][j][k]
        r -=1
  return output1, output2

def mutate (m):
  # could include a constant to control 
  # how much the weight is mutated by
  for i in range(len(m)):
    for j in range(len(m[i])):
      for k in range(len(m[i][j])):
        if random.random() < mutation_rate:
            m[i][j][k] = random.uniform(-2.0,2.0)
  
if __name__ == "__main__":
    main()
