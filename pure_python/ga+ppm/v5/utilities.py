import random
import math

from pylab import zeros, where, array

def crossover(m1, m2, NN):
  # Maybe could be sped up using flatten/reshape output?
  r = random.randint(0, NN.ni*NN.nh + NN.nh*NN.no) 
  output1 = [zeros(m1[0].shape), zeros(m1[1].shape)]
  output2 = [zeros(m1[0].shape), zeros(m1[1].shape)]
  for i in range(len(m1)):
    for j in range(len(m1[i])):
      for k in range(len(m1[i][j])):
        if r >= 0:
          output1[i][j][k] = m1[i][j][k]
          output2[i][j][k] = m2[i][j][k]
        elif r < 0:
          output1[i][j][k] = m2[i][j][k]
          output2[i][j][k] = m1[i][j][k]
        r -= 1
  return output1, output2


def mutate(m, mutation_rate):
  # Variation: could include a constant to control 
  # how much the weight is mutated by
  for i in range(len(m)):
    for j in range(len(m[i])):
      for k in range(len(m[i][j])):
        if random.random() < mutation_rate:
            m[i][j][k] = random.uniform(-2.0,2.0)


def percentAcc(all_aos, targets):
  correct = 0
  for i, trg in enumerate(targets):
    sample_res = where(trg == array(all_aos[i]), True, False)
    if sample_res.all():
      correct += 1
  total = len(all_aos)
  return float(correct) / total

def sigmoid(x):
  return math.tanh(x)
  

def randomizeMatrix(matrix, a, b):
  for i in range(len(matrix)):
    for j in range(len(matrix[0])):
      matrix[i][j] = random.uniform(a, b)


def roulette(fitnessScores):
  cumalativeFitness = 0.0
  r = random.random()
  for i in range(len(fitnessScores)): 
    cumalativeFitness += fitnessScores[i]
    if cumalativeFitness > r: 
      return i
      

def calcFit(numbers):  # each fitness is a fraction of the total error
  total, fitnesses = sum(numbers), []
  for i in range(len(numbers)):           
    try:
      fitness = numbers[i] / total
    except ZeroDivisionError:
      print 0, 'correct'
      fitness = 0
    fitnesses.append(fitness)
  return fitnesses

