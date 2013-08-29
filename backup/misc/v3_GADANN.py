from operator import itemgetter, attrgetter
import math
import random
import string
import GADANN_test as tst
from training_patterns import pat

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
    matches = []
    for j in range(pop_size):
      if errors[i] == errors[j]:
        if j != i:
          matches.append(j)
    #print str(matches).rjust(10)
  return zip(weights, fitnesses)            # weights become item[0] and fitnesses[1] in this way fitness is paired with its weight in a tuple
  
def rankPop (newpopW,pop):
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
  return rankedPop

def iteratePop (rankedPop):
  rankedWeights = [ item[0] for item in rankedPop]
  fitnessScores = [ item[1] for item in rankedPop]
  newpopW = [] # the weights for the new pop
  newpopW = [ eval(repr(x)) for x in rankedWeights[:int(pop_size*0.15)] ]
  #print 'rank[3]', [ [ [round(rankedWeights[3][x][y][z],2) for z in range(len(rankedWeights[3][x][y]))] for y in range(len(rankedWeights[3][x])) ] for x in range(len(rankedWeights[3])) ], 'newp[3]', [ [ [round(newpopW[3][x][y][z],2) for z in range(len(newpopW[3][x][y]))] for y in range(len(newpopW[3][x])) ] for x in range(len(newpopW[3])) ]
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
  #print 'rank[3]', [ [ [round(rankedWeights[3][x][y][z],2) for z in range(len(rankedWeights[3][x][y]))] for y in range(len(rankedWeights[3][x])) ] for x in range(len(rankedWeights[3])) ], 'newp[3]', [ [ [round(newpopW[3][x][y][z],2) for z in range(len(newpopW[3][x][y]))] for y in range(len(newpopW[3][x])) ] for x in range(len(newpopW[3])) ]
  tst.dezip(rankedPop, rankedWeights, fitnessScores)
  tst.elitism(rankedWeights, newpopW, pop_size)
  return newpopW

graphical_error_scale = 1000
max_iterations = 1000
pop_size = 101
mutation_rate = 0.5
crossover_rate = 0.8
ni, nh, no = 4,6,3

def main ():
  # Rank first random population
  pop = [ NN(ni,nh,no) for i in range(pop_size) ] # fresh pop
  pairedPop = pairPop(pop)
  rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True) # THIS IS CORRECT
  tst.correctlyranked(rankedPop)
  
  # Keep iterating new pops until max_iterations
  iters = 0
  while iters != max_iterations:
    print 'Iteration'.rjust(150), iters
    newpopW = iteratePop(rankedPop)
    rankedPop = rankPop(newpopW,pop)
    iters+=1
  
  # test a NN with the fittest weights
  tester = NN (ni,nh,no)
  fittestWeights = [ x[0] for x in rankedPop ]
  tester.assignWeights(fittestWeights, 0)
  tester.test(pat)
  
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

"""" FOR TESTING USING PYTHON EMULATOR

a = [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], [[6.0], [7.0]]]
b = [[[7.0, 6.0], [5.0, 4.0], [3.0, 2.0]], [[1.0], [0.0]]]

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

def assignWeights (wi,wo, weights, I):
  io = 0
  for i in range(ni):
    for j in range(nh):
      wi[I][i][j] = weights[I][io][i][j]
  io = 1
  for j in range(nh):
    for k in range(no):
      wo[I][j][k] = weights[I][io][j][k]

"""
