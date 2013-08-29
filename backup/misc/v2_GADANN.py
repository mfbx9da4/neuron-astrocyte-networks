from operator import itemgetter, attrgetter
import math
import random
import string

#pat = [
    #[[0,0], [0]],
    #[[0,1], [1]],
    #[[1,0], [1]],
    #[[1,1], [0]]
#]

pat = [
  [[5.1, 3.5, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[4.9, 3.0, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[4.7, 3.2, 1.3, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[4.6, 3.1, 1.5, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.0, 3.6, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.4, 3.9, 1.7, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  [[4.6, 3.4, 1.4, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  [[5.0, 3.4, 1.5, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[4.4, 2.9, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[4.9, 3.1, 1.5, 0.1], [1, 0, 0], ['Iris-setosa']] ,
  [[5.4, 3.7, 1.5, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[4.8, 3.4, 1.6, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[4.8, 3.0, 1.4, 0.1], [1, 0, 0], ['Iris-setosa']] ,
  [[4.3, 3.0, 1.1, 0.1], [1, 0, 0], ['Iris-setosa']] ,
  [[5.8, 4.0, 1.2, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.7, 4.4, 1.5, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  [[5.4, 3.9, 1.3, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  [[5.1, 3.5, 1.4, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  [[5.7, 3.8, 1.7, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  [[5.1, 3.8, 1.5, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  [[5.4, 3.4, 1.7, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.1, 3.7, 1.5, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  [[4.6, 3.6, 1.0, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.1, 3.3, 1.7, 0.5], [1, 0, 0], ['Iris-setosa']] ,
  [[4.8, 3.4, 1.9, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.0, 3.0, 1.6, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.0, 3.4, 1.6, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  [[5.2, 3.5, 1.5, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.2, 3.4, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[4.7, 3.2, 1.6, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[4.8, 3.1, 1.6, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.4, 3.4, 1.5, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  [[5.2, 4.1, 1.5, 0.1], [1, 0, 0], ['Iris-setosa']] ,
  [[5.5, 4.2, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[4.9, 3.1, 1.5, 0.1], [1, 0, 0], ['Iris-setosa']] ,
  [[5.0, 3.2, 1.2, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.5, 3.5, 1.3, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[4.9, 3.1, 1.5, 0.1], [1, 0, 0], ['Iris-setosa']] ,
  [[4.4, 3.0, 1.3, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.1, 3.4, 1.5, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.0, 3.5, 1.3, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  [[4.5, 2.3, 1.3, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  [[4.4, 3.2, 1.3, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.0, 3.5, 1.6, 0.6], [1, 0, 0], ['Iris-setosa']] ,
  [[5.1, 3.8, 1.9, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  [[4.8, 3.0, 1.4, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  [[5.1, 3.8, 1.6, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[4.6, 3.2, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.3, 3.7, 1.5, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[5.0, 3.3, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  [[7.0, 3.2, 4.7, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.4, 3.2, 4.5, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.9, 3.1, 4.9, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.5, 2.3, 4.0, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.5, 2.8, 4.6, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.7, 2.8, 4.5, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.3, 3.3, 4.7, 1.6], [0, 1, 0], ['Iris-versicolor']] ,
  [[4.9, 2.4, 3.3, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.6, 2.9, 4.6, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.2, 2.7, 3.9, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.0, 2.0, 3.5, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.9, 3.0, 4.2, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.0, 2.2, 4.0, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.1, 2.9, 4.7, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.6, 2.9, 3.6, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.7, 3.1, 4.4, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.6, 3.0, 4.5, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.8, 2.7, 4.1, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.2, 2.2, 4.5, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.6, 2.5, 3.9, 1.1], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.9, 3.2, 4.8, 1.8], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.1, 2.8, 4.0, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.3, 2.5, 4.9, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.1, 2.8, 4.7, 1.2], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.4, 2.9, 4.3, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.6, 3.0, 4.4, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.8, 2.8, 4.8, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.7, 3.0, 5.0, 1.7], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.0, 2.9, 4.5, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.7, 2.6, 3.5, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.5, 2.4, 3.8, 1.1], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.5, 2.4, 3.7, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.8, 2.7, 3.9, 1.2], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.0, 2.7, 5.1, 1.6], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.4, 3.0, 4.5, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.0, 3.4, 4.5, 1.6], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.7, 3.1, 4.7, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.3, 2.3, 4.4, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.6, 3.0, 4.1, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.5, 2.5, 4.0, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.5, 2.6, 4.4, 1.2], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.1, 3.0, 4.6, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.8, 2.6, 4.0, 1.2], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.0, 2.3, 3.3, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.6, 2.7, 4.2, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.7, 3.0, 4.2, 1.2], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.7, 2.9, 4.2, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.2, 2.9, 4.3, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.1, 2.5, 3.0, 1.1], [0, 1, 0], ['Iris-versicolor']] ,
  [[5.7, 2.8, 4.1, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  [[6.3, 3.3, 6.0, 2.5], [0, 0, 1], ['Iris-virginica']] ,
  [[5.8, 2.7, 5.1, 1.9], [0, 0, 1], ['Iris-virginica']] ,
  [[7.1, 3.0, 5.9, 2.1], [0, 0, 1], ['Iris-virginica']] ,
  [[6.3, 2.9, 5.6, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  [[6.5, 3.0, 5.8, 2.2], [0, 0, 1], ['Iris-virginica']] ,
  [[7.6, 3.0, 6.6, 2.1], [0, 0, 1], ['Iris-virginica']] ,
  [[4.9, 2.5, 4.5, 1.7], [0, 0, 1], ['Iris-virginica']] ,
  [[7.3, 2.9, 6.3, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  [[6.7, 2.5, 5.8, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  [[7.2, 3.6, 6.1, 2.5], [0, 0, 1], ['Iris-virginica']] ,
  [[6.5, 3.2, 5.1, 2.0], [0, 0, 1], ['Iris-virginica']] ,
  [[6.4, 2.7, 5.3, 1.9], [0, 0, 1], ['Iris-virginica']] ,
  [[6.8, 3.0, 5.5, 2.1], [0, 0, 1], ['Iris-virginica']] ,
  [[5.7, 2.5, 5.0, 2.0], [0, 0, 1], ['Iris-virginica']] ,
  [[5.8, 2.8, 5.1, 2.4], [0, 0, 1], ['Iris-virginica']] ,
  [[6.4, 3.2, 5.3, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  [[6.5, 3.0, 5.5, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  [[7.7, 3.8, 6.7, 2.2], [0, 0, 1], ['Iris-virginica']] ,
  [[7.7, 2.6, 6.9, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  [[6.0, 2.2, 5.0, 1.5], [0, 0, 1], ['Iris-virginica']] ,
  [[6.9, 3.2, 5.7, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  [[5.6, 2.8, 4.9, 2.0], [0, 0, 1], ['Iris-virginica']] ,
  [[7.7, 2.8, 6.7, 2.0], [0, 0, 1], ['Iris-virginica']] ,
  [[6.3, 2.7, 4.9, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  [[6.7, 3.3, 5.7, 2.1], [0, 0, 1], ['Iris-virginica']] ,
  [[7.2, 3.2, 6.0, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  [[6.2, 2.8, 4.8, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  [[6.1, 3.0, 4.9, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  [[6.4, 2.8, 5.6, 2.1], [0, 0, 1], ['Iris-virginica']] ,
  [[7.2, 3.0, 5.8, 1.6], [0, 0, 1], ['Iris-virginica']] ,
  [[7.4, 2.8, 6.1, 1.9], [0, 0, 1], ['Iris-virginica']] ,
  [[7.9, 3.8, 6.4, 2.0], [0, 0, 1], ['Iris-virginica']] ,
  [[6.4, 2.8, 5.6, 2.2], [0, 0, 1], ['Iris-virginica']] ,
  [[6.3, 2.8, 5.1, 1.5], [0, 0, 1], ['Iris-virginica']] ,
  [[6.1, 2.6, 5.6, 1.4], [0, 0, 1], ['Iris-virginica']] ,
  [[7.7, 3.0, 6.1, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  [[6.3, 3.4, 5.6, 2.4], [0, 0, 1], ['Iris-virginica']] ,
  [[6.4, 3.1, 5.5, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  [[6.0, 3.0, 4.8, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  [[6.9, 3.1, 5.4, 2.1], [0, 0, 1], ['Iris-virginica']] ,
  [[6.7, 3.1, 5.6, 2.4], [0, 0, 1], ['Iris-virginica']] ,
  [[6.9, 3.1, 5.1, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  [[5.8, 2.7, 5.1, 1.9], [0, 0, 1], ['Iris-virginica']] ,
  [[6.8, 3.2, 5.9, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  [[6.7, 3.3, 5.7, 2.5], [0, 0, 1], ['Iris-virginica']] ,
  [[6.7, 3.0, 5.2, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  [[6.3, 2.5, 5.0, 1.9], [0, 0, 1], ['Iris-virginica']] ,
  [[6.5, 3.0, 5.2, 2.0], [0, 0, 1], ['Iris-virginica']] ,
  [[6.2, 3.4, 5.4, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  [[5.9, 3.0, 5.1, 1.8], [0, 0, 1], ['Iris-virginica']]
]
graphical_error_scale = 1000
max_iterations = 1000
pop_size = 100
mutation_rate = 0.01
crossover_rate = 0.8
ni, nh, no = 4,5,3

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
    # number of nodes in layers
    self.ni = NI + 1 # +1 for bias
    self.nh = NH
    self.no = NO
    
    # initialize node-activations
    self.ai, self.ah, self.ao = [],[], []
    self.ai = [1.0]*self.ni
    self.ah = [1.0]*self.nh
    self.ao = [1.0]*self.no

    # create node weight matrices
    self.wi = [ [0.0]*self.nh for i in range(self.ni) ]
    self.wo = [ [0.0]*self.no for j in range(self.nh) ]
    # initialize node weights to random vals
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
    for p in patterns:
      inputs = p[0]
      rounded = [ round(i) for i in self.runNN(inputs) ]
      if rounded == p[1]: result = '+++++'
      else: result = '-----'
      print '%s %s %s %s %s %s %s' %( 'Inputs:', p[0], '-->', str(self.runNN(inputs)).rjust(65), 'Target', p[1], result)
  
  def sumErrors (self):
    # calculate errors
    error = 0.0
    for p in pat:
      inputs = p[0]
      targets = p[1]
      self.runNN(inputs)
      error += self.calcError(targets)
    inverr = 1.0/error
    print '1/sum(MSEs)', inverr, int(inverr*graphical_error_scale)*'-'.rjust(1)
    return inverr
      
  def calcError (self, targets):
    # calc mean squared error
    # 1/2 for differential convenience & **2 for modulus
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
          same.append(('I',i,j, self.wi[i][j] - weights[I][io][i][j]))
    
    io = 1
    for j in range(self.nh):
      for k in range(self.no):
        if not self.wo[j][k] !=  weights[I][io][j][k]:
          same.append(('O',i,j))
    return same

def roulette (fitnessScores):
  cumalativeFitness = 0.0
  r = random.random()
  for i in range(len(fitnessScores)): 
    cumalativeFitness += fitnessScores[i]
    if cumalativeFitness > r: 
      return i
      
def calcFit (numbers):
  total, fitnesses = sum(numbers), []
  for i in range(len(numbers)):
    fitnesses.append(numbers[i]/total)
  return fitnesses

def pairPop (pop):
  weights, errors = [], []
  for i in range(pop_size):
    print str(i+1).zfill(2),
    weights.append([pop[i].wi,pop[i].wo])
    errors.append(pop[i].sumErrors())
  fitnesses = calcFit(errors)
  return zip(weights, fitnesses)

def rankPop (newpopW):
  pop, errors = [ NN(ni,nh,no) for i in range(pop_size) ], [] # fresh pop
  for i in range(pop_size):
    pop[i].assignWeights(newpopW, i)
  pairedPop = pairPop(pop)
  rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True)  
  return rankedPop

def iteratePop (rankedPop):
  fitnessScores = [ item[1] for item in rankedPop]
  rankedWeights = [ item[0] for item in rankedPop]
  newpopW = [] # the weights for the new pop

  # elitism, top 15% of previous pop is conserved to next generation
  newpopW.extend(rankedWeights[:int(pop_size*0.15)]) 
  
  # BREED NEW POP: select two different chromos,
  # mutate and crossover, repeat until pop_size reached
  while len(newpopW) <= pop_size:  
    index1 = roulette(fitnessScores)
    index2 = roulette(fitnessScores)
    # ensures different chromos are used for breeeding
    while index1 == index2: 
      index2 = roulette(fitnessScores)
    ch1 = rankedWeights[index1] 
    ch2 = rankedWeights[index2]
        
    if random.random() < crossover_rate: 
      ch1, ch2 = crossover(ch1, ch2)
    mutate(ch1)
    mutate(ch2)
    newpopW.append(ch1)
    newpopW.append(ch2)
  return newpopW

def main ():
  # Rank first random population
  pop = [ NN(ni,nh,no) for i in range(pop_size) ] # fresh pop
  pairedPop = pairPop(pop)
  rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True)
  
  # Keep iterating new pops until max_iterations
  iters = 0
  while iters != max_iterations:
    newpopW = iteratePop(rankedPop)
    rankedPop = rankPop(newpopW)
    iters+=1
    print 'Iteration'.rjust(150), iters
  
  # test a NN with the fittest weights
  tester = NN (ni,nh,no)
  fittestWeights = [ x[0] for x in rankedPop ]
  tester.assignWeights(fittestWeights, 0)
  tester.test(pat)

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

"""" FOR TESTING USING PYTHON EMULATOR

a = [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], [[6.0], [7.0]]]

# convert long list into matrix
# for a list of values length 10
# first 8 values are paired into sublists 
def list2sublists(a):
  x,y = 0,1
  I,O = [],[]
  for i in range(5):
     if x < 6: # self.ni * self.nh DONT FORGET THE FUCKING BIAS IN THE INPUT LAYER
         I.append([a[x],a[y]]) # [ i-j1, i-j2]
         x += 2 # self.nh
         y += 2
     else:
       O.append([a[x]])
       x+=1
  return [I,O]

def toFloat (m):
  for i in range(len(m)):
    for j in range(len(m[i])):
      for k in range(len(m[i][j])):
        m[i][j][k] = float(m[i][j][k])
  return m

"""
