from operator import itemgetter, attrgetter
import math
import random
import string
import GADANN_test as tst
from training_patterns import pat
from training_patterns import testpat
import timeit
from timeit import Timer as t
import matplotlib.pyplot as plt
import numpy as np

def sigmoid (x):
  return math.tanh(x)

def makeMatrix ( I, J, fill=0.0):
  m = []
  for i in range(I):
    m.append([fill]*J)
  return m
  
def randomizeMatrix ( matrix, a, b):
  for i in range ( len (matrix) ):
    for j in range ( len (matrix[0]) ):
      matrix[i][j] = random.uniform(a,b)

class NN:
  def __init__(self, NI, NH, NO):
    self.ni = NI + 1 # +1 for bias
    self.nh = NH
    self.no = NO
    self.ai = [1.0]*self.ni
    self.ah = [1.0]*self.nh
    self.ao = [1.0]*self.no
    self.Ah = [ [0]*Athresh ]*self.nh
    self.Ao = [ [0]*Athresh ]*self.no
    self.Ah_countdwn = [0]*self.nh
    self.Ao_countdwn = [0]*self.no
    self.wi = [ [0.0]*self.nh ]*self.ni
    self.wo = [ [0.0]*self.no ]*self.nh
    randomizeMatrix ( self.wi, -0.2, 0.2 )
    randomizeMatrix ( self.wo, -2.0, 2.0 )

  def runNN (self, inputs):
    if len(inputs) != self.ni-1:
      print 'incorrect number of inputs'
    for i in range(self.ni-1):
      self.ai[i] = inputs[i]
    for j in range(self.nh):
      sum = 0.0
      for i in range(self.ni):
        sum += self.ai[i] * self.wi[i][j] 
      self.ah[j] = sigmoid (sum)
    for k in range(self.no):
      sum = 0.0
      for j in range(self.nh):        
        sum += self.ah[j] * self.wo[j][k] 
      self.ao[k] = sigmoid (sum)
    return self.ao
    
  def runNGA (self, inputs):
    if len(inputs) != self.ni-1:
      print 'incorrect number of inputs'
    for i in range(self.ni-1):
      self.ai[i] = inputs[i]
  
    for j in range(self.nh):
      self.ah[j] = sigmoid(sum([ self.ai[i] * self.wi[i][j] for i in range(self.ni) ]))
      # check neuron activation
      if self.ah[j] > 0:
        self.Ah[j] = self.Ah[j][1:] + [1]
      elif self.ah[j] <= 0:
        self.Ah[j] = self.Ah[j][1:] + [-1]
      # check if astrocyte should be activated
      if sum(self.Ah[j]) >= Athresh:
        self.Ah_countdwn[j] = Adur
      elif sum(self.Ah[j]) <= -Athresh:
        self.Ah_countdwn[j] = -Adur
        self.Ah[j] = [ [0]*Athresh ]*self.nh
      # check if astrocyte is active --> perform actions
      if self.Ah_countdwn[j] > 0:
        for I in range(ni):
          for J in range(nh):
            self.wi[I][J] += self.wi[I][J]*0.25
            print self.wi[I][J]
        #print 'self.Ah_countdwn[j]',self.Ah_countdwn[j]
        self.Ah_countdwn[j] -= 1
      elif self.Ah_countdwn[j] < 0:
        for I in range(ni):
          for J in range(nh):
            self.wi[I][J] += self.wi[I][J]*-0.5
            print self.wi[I][J]
        #print 'self.Ah_countdwn[j]',self.Ah_countdwn[j]
        self.Ah_countdwn[j] += 1
    #print 'self.Ah',self.Ah
    
    for k in range(self.no):
      self.ao[k] = sigmoid(sum([ self.ah[j] * self.wo[j][k] for i in range(self.nh) ]))
      # check neuron activation
      if self.ao[k] > 0:
        self.Ah[k] = self.Ah[k][1:] + [1]
      elif self.ao[k] <= 0:
        self.Ah[k] = self.Ah[k][1:] + [-1]
      else:
        self.Ah[k] = self.Ah[k][1:] + [0]
      # check if astrocyte should be activated
      if sum(self.Ah[k]) == Athresh:
        self.Ah_countdwn[k] = Adur
      elif sum(self.Ah[k]) == -Athresh:
        self.Ah_countdwn[k] = -Adur
      # check if astrocyte is active --> perform actions
      if self.Ah_countdwn[k] > 0:
        for J in range(nh):
          for K in range(no):
            self.wo[J][K] += self.wo[J][K]*0.25
        self.Ah_countdwn[k] -= 1
      elif self.Ah_countdwn[k] < 0:
        for J in range(nh):
          for K in range(no):
            self.wo[J][K] += self.wo[J][K]*-0.5
        self.Ah_countdwn[k] += 1
      

  def weights(self):
    print 'Input weights:'
    for i in range(self.ni):
      print self.wi[i]
    print
    print 'Output weights:'
    for j in range(self.nh):
      print self.wo[j]
    print ''

  def test(self, patterns):
    results, targets = [], []
    for p in patterns:
      inputs = p[0]
      rounded = [ round(i) for i in self.runNN(inputs) ]
      if rounded == p[1]: result = '+++++'
      else: result = '-----'
      #print '%s %s %s %s %s %s %s' %( 'Inputs:', p[0], '-->', str(self.runNN(inputs)).rjust(65), 'Target', p[1], result)
      results+= self.runNN(inputs)
      targets += p[1]
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
          same.append(('I',i,j, self.wi[i][j],weights[I][io][i][j],self.wi[i][j] - weights[I][io][i][j]))

    io = 1
    for j in range(self.nh):
      for k in range(self.no):
        if self.wo[j][k] !=  weights[I][io][j][k]:
          same.append(('O',i,j), self.wo[j][k], weights[I][io][j][k])
    if same != []:
      print same

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

# takes a population of NN objects
def pairPop (pop):
  weights, errors = [], []
  for i in range(len(pop)):                 # for each individual
    weights.append([pop[i].wi,pop[i].wo])   # append input & output weights of individual to list of all pop weights
    errors.append(pop[i].sumErrors())       # append 1/sum(MSEs) of individual to list of pop errors
  fitnesses = calcFit(errors)               # fitnesses are a fraction of the total error
  for i in range(int(pop_size*0.15)): 
    print str(i).zfill(2), '1/sum(MSEs)', str(errors[i]).rjust(15), str(int(errors[i]*graphical_error_scale)*'-').rjust(20), 'fitness'.rjust(12), str(fitnesses[i]).rjust(17), str(int(fitnesses[i]*1000)*'-').rjust(20)
  print 
  return zip(weights, errors,fitnesses)            # weights become item[0] and fitnesses[1] in this way fitness is paired with its weight in a tuple
  
def rankPop (newpopW):
  pop, errors, copy = [ NN(ni,nh,no) for i in range(pop_size) ], [], []           # a fresh pop of NN's are assigned to a list of len pop_size
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
  return rankedPop, eval(repr(rankedPop[0][1])), float(sum(errors))/float(len(errors))

def iteratePop (rankedPop):
  rankedWeights = [ item[0] for item in rankedPop]
  fitnessScores = [ item[1] for item in rankedPop]
  newpopW = [] # the weights for the new pop
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
  tst.elitism(rankedWeights, newpopW, pop_size)
  return newpopW

def NGA(newpopW):
  pop = [ NN(ni,nh,no) for i in range(pop_size) ]
  weights = []
  for i in range(pop_size):
    pop[i].assignWeights(newpopW, i)
    for p in range(len(pat)):
      inputs, targets = pat[p][0], pat[p][1]
      for m in range(m_iters):
        pop[i].runNGA(inputs)
        #print 'pop['+str(i)+']','pat',p,'m',m
    weights.append([pop[i].wi,pop[i].wo])
  return weights
        

graphical_error_scale = 500
max_iterations = 10
pop_size = 150
mutation_rate = 0.1
crossover_rate = 0.8
ni, nh, no = 4,6,1
m_iters = 6
Athresh = 2
Adur = 4

def main ():
  # Rank first random population
  pop = [ NN(ni,nh,no) for i in range(pop_size) ] # fresh pop
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
  plt.title('Top individual error evolution')
  plt.title('Population average error evolution')
  plt.plot( avgs, '-g', linewidth = 0.5)
  plt.plot( tops, '-r', linewidth = 2)
  plt.ylabel('Inverse error')
  plt.xlabel('Iterations')
  
  plt.show()
  
  print 'max_iterations',max_iterations,'\tpop_size',pop_size,'pop_size*0.15',int(pop_size*0.15),'\tmutation_rate',mutation_rate,'crossover_rate',crossover_rate,'ni, nh, no',ni, nh, no
def crossover (m1, m2):
  r = random.randint(0,8) # 2*2 input + 1*2 bias + 2*1 hidden = total weights
  output1 = [ [ [0.0]*(nh) for i in range(ni+1) ] , [ [0.0]*(no) for j in range(nh) ]]
  output2 = [ [ [0.0]*(nh) for i in range(ni+1) ] , [ [0.0]*(no) for j in range(nh) ]]
  del i, j
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
  return m
  
if __name__ == "__main__":
    main()
