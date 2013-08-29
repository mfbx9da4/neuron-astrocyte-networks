from decimal import *
from operator import itemgetter, attrgetter
import math
import random
import string

def sigmoid (x):
  return math.tanh(x)
  
def dsigmoid (y):
  return 1 - y**2

def makeMatrix ( I, J, fill=Decimal(0.0)):
  m = []
  for i in range(I):
    m.append([fill]*J)
  return m
  
def randomizeMatrix ( matrix, a, b):
  for i in range ( len (matrix) ):
    for j in range ( len (matrix[0]) ):
      matrix[i][j] = Decimal(random.uniform(a,b))

class NN:
  def __init__(self, NI, NH, NO):
    # number of nodes in layers
    self.ni = NI + 1 # +1 for bias
    self.nh = NH
    self.no = NO
    
    # initialize node-activations
    self.ai, self.ah, self.ao = [],[], []
    self.ai = [Decimal(1.0)]*self.ni
    self.ah = [Decimal(1.0)]*self.nh
    self.ao = [Decimal(1.0)]*self.no

    # create node weight matrices
    self.wi = makeMatrix (self.ni, self.nh)
    self.wo = makeMatrix (self.nh, self.no)
    # initialize node weights to random vals
    randomizeMatrix ( self.wi, -0.2, 0.2 )
    randomizeMatrix ( self.wo, -2.0, 2.0 )
    # create last change in weights matrices for momentum
    self.ci = makeMatrix (self.ni, self.nh)
    self.co = makeMatrix (self.nh, self.no)
    
  def runNN (self, inputs):
    if len(inputs) != self.ni-1:
      print 'incorrect number of inputs'
    for i in range(self.ni-1):
      self.ai[i] = Decimal(inputs[i])
    for j in range(self.nh):
      sum = Decimal(0.0)
      for i in range(self.ni):
        sum += self.ai[i] * self.wi[i][j] 
      self.ah[j] = Decimal(sigmoid (sum))
    for k in range(self.no):
      sum = Decimal(0.0)
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
      print 'Inputs:', p[0], '-->', self.runNN(inputs), '\tTarget', p[1]
  
  def train (self, patterns, max_iterations = 1000, N=0.5, M=0.1):
    for i in range(max_iterations):
      for p in patterns:
        inputs = p[0]
        targets = p[1]
        self.runNN(inputs)
        error = self.backPropagate(targets, N, M)
      if i % 50 == 0:
        print 'Combined error', error
    self.test(patterns)
  
  def sumErrors (self):
    # calculate errors
    error = Decimal(0.0)
    for p in pat:
      inputs = p[0]
      targets = p[1]
      self.runNN(inputs)
      error += self.calcError(targets)
    #print 'MSE', error
    inverr = Decimal(1.0)/error
    # for each instance, over the errors of all ouput nodes
    # best networks will have lowest error
    # invert error so now best NNs have largest values
    # take each error as a percentage of total error
    # now the largest fitness has the most chance of being picked in roulette
    
    return inverr
      
  def calcError (self, targets):
    # calc combined mean squared error
    # 1/2 for differential convenience & **2 for modulus
    error = Decimal(0.0)
    for k in range(len(targets)):
      error += Decimal(0.5) * (Decimal(targets[k])-Decimal(self.ao[k]))**2
    return error
  
  def assignWeights (self, weights, I):
    io = 0
    for i in range(self.ni):
      for j in range(self.nh):
        self.wi[i][j] = weights[I][io][i][j]
    del j
     
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


pat = [
    [[0,0], [0]],
    [[0,1], [1]],
    [[1,0], [1]],
    [[1,1], [0]]
]
pop_size = 30

def calcFit (numbers):
  total, fitnesses = sum(numbers), []
  for i in range(len(numbers)):
    fitnesses.append(numbers[i]/total)
  return fitnesses

def generatePop ( In, Hn, On):
  pop = []
  for i in range(pop_size):
    pop.append(NN(2, 2, 1))
  return pop

def pairPop (pop):    
  weights, errors = [], []
  for i in range(pop_size):
    weights.append([pop[i].wi,pop[i].wo])
    errors.append(pop[i].sumErrors())
  fitnesses = calcFit(errors)
  #for i in range(pop_size):
    #print str(i).rjust(2), errors[i], '\t', fitnesses[i]
  return zip(weights, fitnesses)
  
def roulette (fitnessScores):
  cumalativeFitness = Decimal(0.0)
  r = random.random()
  for i in range(len(fitnessScores)): 
    cumalativeFitness += fitnessScores[i]
    if cumalativeFitness > r: 
      return i

def crossover (m1, m2):
  r = random.randint(0,8) # 2*2 input + 1*2 bias + 2*1 hidden = total weights
  ni,nh,no = 2+1,2,1
  output1 = [ [ [Decimal(0.0)]*(nh) for i in range(ni) ] , [ [Decimal(0.0)]*(no) for j in range(nh) ]]
  output2 = [ [ [Decimal(0.0)]*(nh) for i in range(ni) ] , [ [Decimal(0.0)]*(no) for j in range(nh) ]]
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
  #print 'crossed1',output1
  #print 'crossed2', output2
  return output1, output2

# use regexp to extract everything aprt from decimal point and '-'
# 
def mutate (m):
  mutation_rate = 0.2
  for i in range(len(m)):
    for j in range(len(m[i])):
      for k in range(len(m[i][j])):
        temp = [ x for x in m[i][j][k].as_tuple()[1] ]
        exp = 0
        while len(temp) < 55:
          temp.append(0)
          exp += 1
        for x in range(len(temp)):
          if random.random() < mutation_rate:
            temp[x]= random.randint(0,9)
        #print 'aft',temp
        m[i][j][k] = Decimal( (m[i][j][k].as_tuple()[0],temp,m[i][j][k].as_tuple()[2]-exp)  )
  return m

def rankPop (newpopW):
  pop = generatePop(2, 2, 1)
  errors = []
  for i in range(pop_size):
    pop[i].assignWeights(newpopW, i)
  pairedPop = pairPop(pop)
  rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True)  
  return rankedPop

def iteratePop (rankedPop):
  crossover_rate = 0.8
  fitnessScores = [ item[1] for item in rankedPop]
  rankedWeights = [ item[0] for item in rankedPop]
  newpopW = []

  # elitism
  newpopW.extend(rankedWeights[:int(pop_size*0.15)]) # check format is ok
  #for i in range(int(pop_size*0.15)):
    #newpopW.append([pop[i].wi,pop[i].wo])
  
  # breed
  while len(newpopW) != pop_size:  
    ch1 = rankedWeights[roulette(fitnessScores)] # could check that indexes are not the same
    ch2 = rankedWeights[roulette(fitnessScores)]
    if random.random() < crossover_rate: # rate dependent crossover of selected chromosomes
      ch1, ch2 = crossover(ch1, ch2)
    mutate(ch1)
    mutate(ch2)
    newpopW.append(ch1)
    newpopW.append(ch2)
  return newpopW

def main ():
  pop = generatePop(2, 2, 1)

  # rank Pop
  pairedPop = pairPop(pop)
  rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True)
  i = 0
  for j, k in rankedPop: print 'ranked', i, k,'\t'; i +=1
  
  iters, max_iterations = 0, 1000
  while iters != max_iterations:
    newpopW = iteratePop(rankedPop)
    rankedPop = rankPop(newpopW)
    iters+=1
    print '\t\t\t\t\tIteration', iters
    
  tester = NN (2,2,1)
  rankedWeights = [ x[0] for x in rankedPop ]
  tester.assignWeights(rankedWeights, 0)
  tester.test(pat)
    # assign weights of top NN
    # test
  
  

  

  
if __name__ == "__main__":
    main()

"""" 
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
  
a = [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], [[6.0], [7.0]]]
a = [[[Decimal('0.011'), Decimal('1.01')], [Decimal('2.01'), Decimal('3.01')], [Decimal('4.01'), Decimal('5.01')]], [[Decimal('6.01')], [Decimal('7.01')]]]

  

def toFloat (m):
  for i in range(len(m)):
    for j in range(len(m[i])):
      for k in range(len(m[i][j])):
        m[i][j][k] = float(m[i][j][k])
  return m
  
def toDecimal (m):
  for i in range(len(m)):
    for j in range(len(m[i])):
      for k in range(len(m[i][j])):
        m[i][j][k] = Decimal(m[i][j][k])
  return m

"""


