from operator import itemgetter, attrgetter
import math
import random
import string
import GADANN_test

pat = [
    [[0,0], [0]],
    [[0,1], [1]],
    [[1,0], [1]],
    [[1,1], [0]]
]

#pat = [
  #[[5.1, 3.5, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.9, 3.0, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.7, 3.2, 1.3, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.6, 3.1, 1.5, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.0, 3.6, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.4, 3.9, 1.7, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.6, 3.4, 1.4, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.0, 3.4, 1.5, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.4, 2.9, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.9, 3.1, 1.5, 0.1], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.4, 3.7, 1.5, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.8, 3.4, 1.6, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.8, 3.0, 1.4, 0.1], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.3, 3.0, 1.1, 0.1], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.8, 4.0, 1.2, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.7, 4.4, 1.5, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.4, 3.9, 1.3, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.1, 3.5, 1.4, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.7, 3.8, 1.7, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.1, 3.8, 1.5, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.4, 3.4, 1.7, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.1, 3.7, 1.5, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.6, 3.6, 1.0, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.1, 3.3, 1.7, 0.5], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.8, 3.4, 1.9, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.0, 3.0, 1.6, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.0, 3.4, 1.6, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.2, 3.5, 1.5, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.2, 3.4, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.7, 3.2, 1.6, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.8, 3.1, 1.6, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.4, 3.4, 1.5, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.2, 4.1, 1.5, 0.1], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.5, 4.2, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.9, 3.1, 1.5, 0.1], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.0, 3.2, 1.2, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.5, 3.5, 1.3, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.9, 3.1, 1.5, 0.1], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.4, 3.0, 1.3, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.1, 3.4, 1.5, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.0, 3.5, 1.3, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.5, 2.3, 1.3, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.4, 3.2, 1.3, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.0, 3.5, 1.6, 0.6], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.1, 3.8, 1.9, 0.4], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.8, 3.0, 1.4, 0.3], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.1, 3.8, 1.6, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[4.6, 3.2, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.3, 3.7, 1.5, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[5.0, 3.3, 1.4, 0.2], [1, 0, 0], ['Iris-setosa']] ,
  #[[7.0, 3.2, 4.7, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.4, 3.2, 4.5, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.9, 3.1, 4.9, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.5, 2.3, 4.0, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.5, 2.8, 4.6, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.7, 2.8, 4.5, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.3, 3.3, 4.7, 1.6], [0, 1, 0], ['Iris-versicolor']] ,
  #[[4.9, 2.4, 3.3, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.6, 2.9, 4.6, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.2, 2.7, 3.9, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.0, 2.0, 3.5, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.9, 3.0, 4.2, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.0, 2.2, 4.0, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.1, 2.9, 4.7, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.6, 2.9, 3.6, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.7, 3.1, 4.4, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.6, 3.0, 4.5, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.8, 2.7, 4.1, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.2, 2.2, 4.5, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.6, 2.5, 3.9, 1.1], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.9, 3.2, 4.8, 1.8], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.1, 2.8, 4.0, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.3, 2.5, 4.9, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.1, 2.8, 4.7, 1.2], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.4, 2.9, 4.3, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.6, 3.0, 4.4, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.8, 2.8, 4.8, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.7, 3.0, 5.0, 1.7], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.0, 2.9, 4.5, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.7, 2.6, 3.5, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.5, 2.4, 3.8, 1.1], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.5, 2.4, 3.7, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.8, 2.7, 3.9, 1.2], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.0, 2.7, 5.1, 1.6], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.4, 3.0, 4.5, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.0, 3.4, 4.5, 1.6], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.7, 3.1, 4.7, 1.5], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.3, 2.3, 4.4, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.6, 3.0, 4.1, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.5, 2.5, 4.0, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.5, 2.6, 4.4, 1.2], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.1, 3.0, 4.6, 1.4], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.8, 2.6, 4.0, 1.2], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.0, 2.3, 3.3, 1.0], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.6, 2.7, 4.2, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.7, 3.0, 4.2, 1.2], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.7, 2.9, 4.2, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.2, 2.9, 4.3, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.1, 2.5, 3.0, 1.1], [0, 1, 0], ['Iris-versicolor']] ,
  #[[5.7, 2.8, 4.1, 1.3], [0, 1, 0], ['Iris-versicolor']] ,
  #[[6.3, 3.3, 6.0, 2.5], [0, 0, 1], ['Iris-virginica']] ,
  #[[5.8, 2.7, 5.1, 1.9], [0, 0, 1], ['Iris-virginica']] ,
  #[[7.1, 3.0, 5.9, 2.1], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.3, 2.9, 5.6, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.5, 3.0, 5.8, 2.2], [0, 0, 1], ['Iris-virginica']] ,
  #[[7.6, 3.0, 6.6, 2.1], [0, 0, 1], ['Iris-virginica']] ,
  #[[4.9, 2.5, 4.5, 1.7], [0, 0, 1], ['Iris-virginica']] ,
  #[[7.3, 2.9, 6.3, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.7, 2.5, 5.8, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  #[[7.2, 3.6, 6.1, 2.5], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.5, 3.2, 5.1, 2.0], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.4, 2.7, 5.3, 1.9], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.8, 3.0, 5.5, 2.1], [0, 0, 1], ['Iris-virginica']] ,
  #[[5.7, 2.5, 5.0, 2.0], [0, 0, 1], ['Iris-virginica']] ,
  #[[5.8, 2.8, 5.1, 2.4], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.4, 3.2, 5.3, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.5, 3.0, 5.5, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  #[[7.7, 3.8, 6.7, 2.2], [0, 0, 1], ['Iris-virginica']] ,
  #[[7.7, 2.6, 6.9, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.0, 2.2, 5.0, 1.5], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.9, 3.2, 5.7, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  #[[5.6, 2.8, 4.9, 2.0], [0, 0, 1], ['Iris-virginica']] ,
  #[[7.7, 2.8, 6.7, 2.0], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.3, 2.7, 4.9, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.7, 3.3, 5.7, 2.1], [0, 0, 1], ['Iris-virginica']] ,
  #[[7.2, 3.2, 6.0, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.2, 2.8, 4.8, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.1, 3.0, 4.9, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.4, 2.8, 5.6, 2.1], [0, 0, 1], ['Iris-virginica']] ,
  #[[7.2, 3.0, 5.8, 1.6], [0, 0, 1], ['Iris-virginica']] ,
  #[[7.4, 2.8, 6.1, 1.9], [0, 0, 1], ['Iris-virginica']] ,
  #[[7.9, 3.8, 6.4, 2.0], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.4, 2.8, 5.6, 2.2], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.3, 2.8, 5.1, 1.5], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.1, 2.6, 5.6, 1.4], [0, 0, 1], ['Iris-virginica']] ,
  #[[7.7, 3.0, 6.1, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.3, 3.4, 5.6, 2.4], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.4, 3.1, 5.5, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.0, 3.0, 4.8, 1.8], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.9, 3.1, 5.4, 2.1], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.7, 3.1, 5.6, 2.4], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.9, 3.1, 5.1, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  #[[5.8, 2.7, 5.1, 1.9], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.8, 3.2, 5.9, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.7, 3.3, 5.7, 2.5], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.7, 3.0, 5.2, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.3, 2.5, 5.0, 1.9], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.5, 3.0, 5.2, 2.0], [0, 0, 1], ['Iris-virginica']] ,
  #[[6.2, 3.4, 5.4, 2.3], [0, 0, 1], ['Iris-virginica']] ,
  #[[5.9, 3.0, 5.1, 1.8], [0, 0, 1], ['Iris-virginica']]
#]

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
    self.wi = [ [0.0]*self.nh for i in range(self.ni) ]
    self.wo = [ [0.0]*self.no for j in range(self.nh) ]
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

  def test(self, patterns):
    for p in patterns:
      inputs = p[0]
      rounded = [ round(i) for i in self.runNN(inputs) ]
      if rounded == p[1]: result = '+++++'
      else: result = '-----'
      print '%s %s %s %s %s %s %s' %( 'Inputs:', p[0], '-->', str(self.runNN(inputs)).rjust(65), 'Target', p[1], result)

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
  return zip(weights, errors, fitnesses)            # weights become item[0] and fitnesses[1] in this way fitness is paired with its weight in a tuple
  
def rankPop (pop, rankedPop):
  newpopW = [ x[0] for x in rankedPop ]
  errors = [ x[1] for x in rankedPop ]
  #pop = [ NN(ni,nh,no) for i in range(pop_size) ]; print 'fresh pop'
  for i in range(pop_size):              
    #print 'newpopW['+str(i)+']',[ round(x[0],2) for x in newpopW[i][1]]
    print i,'pop',[ round(x[0],2) for x in pop[i].wo], '\tpopW',i,[ round(x[0],2) for x in newpopW[i][1]], '\tbefore assigning weights', errors[i]
    pop[i].assignWeights(newpopW, i)                                    # each individual is assigned the weights generated from previous iteration
    print i,'pop',[ round(x[0],2) for x in pop[i].wo], '\tpopW',i,[ round(x[0],2) for x in newpopW[i][1]], '\tafter  assigning weights', chr(219)*int(errors[i]*70.00)
  print '-'*40,'end assigning weights loop', 'len newpopW', len(newpopW)
  for i in range(pop_size):              
    print i,'pop',[ round(x[0],2) for x in pop[i].wo], '\tpopW',i,[ round(x[0],2) for x in newpopW[i][1]], '\t',
    matches = []
    for j in range(pop_size):
      if [ x[0] for x in pop[j].wo] == [ x[0] for x in newpopW[i][1]]:
        matches += str(j)
    print matches

  pairedPop = pairPop(pop)                                              # the fitness of these weights is calculated and tupled with the weights
  rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True)   # weights are sorted in descending order of fitness (fittest first)
  return rankedPop


graphical_error_scale = 5
max_iterations = 4
pop_size = 10
mutation_rate = 0.05
crossover_rate = 0.6
ni, nh, no = 2,2,1

def main ():
  #rankedPop = [([[[0.027876847102269176, -0.08644257278865086], [-0.07944970315151734, 0.12341588363158124], [0.14372841569522005, -0.17673296165438496]], [[0.8944483866425483], [-1.721391025527315]]], 1.6617403461588902, 0.15160340558509977), ([[[-0.008981238441592876, -0.19332196081800626], [0.14010296998412025, 0.01595424187021513], [-0.08051054039313615, -0.08042728458046211]], [[0.13683676170390457], [-1.5652780148993655]]], 1.5020080917041694, 0.13703075961601288), ([[[0.0577007411941442, 0.061884138073667105], [-0.11358250389685597, -0.14898294759161718], [-0.06621974670333555, 0.19090619604639397]], [[-1.4512167889270158], [0.32075055239828076]]], 1.3965003918577275, 0.1274051122342518), ([[[0.1135360396465846, 0.16941540126725074], [-0.03421148785124109, -0.023318801389095922], [0.06172174918657913, -0.06184662238779462]], [[1.1584721131076545], [-0.570341874207013]]], 1.2408616260577863, 0.11320592221586324), ([[[-0.05358088524869681, 0.14938977799614567], [0.1499756807006049, -0.07006749143848975], [0.15013470717612964, 0.044217367353700604]], [[0.07699445620878498], [1.0932064615443315]]], 1.2066934681610644, 0.11008870290317939), ([[[-0.026505521293627293, 0.006440324070783154], [0.015139796097942748, 0.07717959998055596], [0.12806688893990825, 0.00206127534424555]], [[0.15692505544776614], [-0.5515191403971786]]], 0.9891404617512051, 0.0902409711301325), ([[[0.18082572918194523, 0.11770523392448412], [-0.1622524503625043, 0.18126898040038297], [-0.08335859483114949, 0.11320956483439776]], [[-0.47721957271113213], [-0.4273271410873658]]], 0.8553687824123537, 0.07803675270004773), ([[[-0.025615365134260243, -0.15362613923018176], [0.07682756672460261, -0.14389956338759965], [-0.16425940749709614, 0.10270250126666686]], [[1.613046883548884], [-1.5943887037556004]]], 0.71028420531042, 0.06480043931488544), ([[[-0.03567341434404442, -0.10317757344244033], [-0.10636546785100083, -0.04443627654565785], [0.18585457055652155, 0.047567145860316185]], [[-1.1965279488412692], [1.6144927793745203]]], 0.7032823728649018, 0.06416165020049072), ([[[-0.14036784910201436, -0.1578002594828386], [-0.0343171902815374, -0.01755673520666759], [0.15529103696218127, -0.04576468540222142]], [[0.3117460755557748], [1.4950669017133351]]], 0.6952219502536421, 0.06342628410003662)]
  pop = [ NN(ni,nh,no) for i in range(pop_size) ] # fresh pop  

  pairedPop = pairPop(pop)
  pairedPop = [([[[0.06632636401012287, -0.09001059900560873], [-0.028750348038027257, 0.14048154600684287], [0.1717564254542408, -0.08212881512568587]], [[-1.1516345660419813], [1.736460334463561]]], 0.5554036442534311, 0.05828240676098087), ([[[-0.0843408457093418, 0.1830634525082253], [-0.028741680836167516, 0.0900736999679872], [-0.12892636808227842, 0.05299997427819125]], [[-1.85448051697978], [1.4057804188287988]]], 1.923562742188421, 0.20185295384799007), ([[[0.025108145586930336, -0.18037034829482057], [-0.14284768602326592, 0.040647741985793295], [-0.13111738764054218, 0.031586165123900056]], [[1.5407300374556003], [-1.0174350253766757]]], 0.6156245452353734, 0.06460180902428467), ([[[0.14442992977996133, -0.10871879827267153], [0.1247103658985173, 0.06738546290954012], [0.02745623267579983, -0.0573065654021164]], [[-0.9801815987513032], [0.9103833562279995]]], 0.6380638038205257, 0.06695651809003564), ([[[0.08022435601519556, 0.11990525674706842], [0.1480503262185328, -0.18360623355189729], [-0.10547276650918574, -0.13584654611643276]], [[0.37059280121460825], [0.7106108687312411]]], 0.7964410249279296, 0.08357615268869445), ([[[0.07404652162309056, 0.13490937438364226], [-0.023843501414605978, 0.04567012234323514], [0.008714214657910307, 0.029754979766083656]], [[-0.47685369834090974], [-1.8580362636923011]]], 0.6238164647507584, 0.06546144469698656), ([[[-0.0883173014435894, 0.013233622447340554], [-0.03228093933582374, 0.18758876187052265], [0.1536265607162487, -0.006551026379533947]], [[1.364985197485816], [-1.9690817042513293]]], 0.8297656154072125, 0.08707313611246621), ([[[0.15156465510710243, -0.11685484972052779], [-0.01062265336399068, -0.19467544774018408], [0.17699253221348715, 0.16845081666390754]], [[0.36259872489074985], [-0.25085329674766044]]], 1.1760709505395326, 0.12341338813369697), ([[[-0.04556919628665673, 0.1444109869586035], [-0.00509101960770203, -0.1390249488359081], [0.12162495023407621, 0.17959522686576285]], [[-0.7195321676040285], [-0.17996839300083645]]], 0.8184859525258074, 0.08588948183330773), ([[[-0.06598583947371037, -0.19223484441876115], [0.0640210053455833, 0.12873909009353518], [-0.052608847490460686, -0.19096178204849765]], [[-1.1411157735737731], [-1.524403900660063]]], 1.5522901184791271, 0.16289270881155685)]
  newpopW = [ x[0] for x in pairedPop ]
  errors = [ x[1] for x in pairedPop ]
  rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True)
  for i in range(pop_size):              
    print 'before while',i,'pop',[ round(x[0],2) for x in pop[i].wo], '\tpopW',i,[ round(x[0],2) for x in newpopW[i][1]], '\tbefore assigning weights', errors[i]
  
  # Keep iterating new pops until max_iterations
  iters = 0
  while iters != max_iterations:
    newpopW = [ x[0] for x in rankedPop ]
    errors = [ x[1] for x in rankedPop ]
    pop = [ NN(ni,nh,no) for i in range(pop_size) ]
    for i in range(pop_size):              
      print 'inwhile loop',i,'pop',[ round(x[0],2) for x in pop[i].wo], '\tpopW',i,[ round(x[0],2) for x in newpopW[i][1]], '\tbefore assigning weights', errors[i]
    for i in range(pop_size):              
      print i,'pop',[ round(x[0],2) for x in pop[i].wo], '\tpopW',i,[ round(x[0],2) for x in newpopW[i][1]], '\tbefore assigning weights', errors[i]
      pop[i].assignWeights(newpopW, i)                                    # each individual is assigned the weights generated from previous iteration
      print i,'pop',[ round(x[0],2) for x in pop[i].wo], '\tpopW',i,[ round(x[0],2) for x in newpopW[i][1]], '\tafter  assigning weights', chr(219)*int(errors[i]*70.00)
    print '-'*40,'end assigning weights loop', 'len newpopW', len(newpopW)
    for i in range(pop_size):              
      print i,'pop',[ round(x[0],2) for x in pop[i].wo], '\tpopW',i,[ round(x[0],2) for x in newpopW[i][1]], '\t',
      matches = []
      for j in range(pop_size):
        if [ x[0] for x in pop[j].wo] == [ x[0] for x in newpopW[i][1]]:
          matches += str(j)
      print matches

    pairedPop = pairPop(pop)                                              # the fitness of these weights is calculated and tupled with the weights
    rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True)   # weights are sorted in descending order of fitness (fittest first)
    iters+=1
    print 'Iteration'.rjust(150), iters
  
  tester = NN (ni,nh,no)
  fittestWeights = [ x[0] for x in rankedPop ]
  tester.assignWeights(fittestWeights, 0)
  tester.test(pat)
  
  print 'max_iterations',max_iterations, 'pop_size',pop_size,'mutation_rate',mutation_rate,'crossover_rate',crossover_rate,'ni, nh, no',ni, nh, no

if __name__ == "__main__":
    main()
