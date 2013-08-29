# takes a population of NN objects
def pairPop (pop):
  weights, errors = [], []
  for i in range(len(pop)):                 # for each individual
    weights.append([list(pop[i].wi), list(pop[i].wo)])   # append input & output weights of individual to list of all pop weights
    errors.append(pop[i].sumErrors())       # append 1/sum(MSEs) of individual to list of pop errors
  fitnesses = calcFit(errors)               # fitnesses are a fraction of the total error
  for i in range(int(pop_size * 0.15)): 
    print str(i).zfill(2), '1/sum(MSEs)', str(errors[i]).rjust(15), str(int(errors[i]*graphical_error_scale)*'-').rjust(20), 'fitness'.rjust(12), str(fitnesses[i]).rjust(17), str(int(fitnesses[i]*1000)*'-').rjust(20)
  del pop
  return zip(weights, errors,fitnesses)            # weights become item[0] and fitnesses[1] in this way fitness is paired with its weight in a tuple
  
def rankPop (newpopW, pop):
  errors, copy = [], []           # a fresh pop of NN's are assigned to a list of len pop_size
  #pop = [NN(ni,nh,no)]*pop_size # this does not work as they are all copies of eachother
  pop = [NN(ni,nh,no) for i in range(pop_size) ]
  for i in range(pop_size): copy.append(list(newpopW[i]))
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
  """
  rankedPop is zip(weights, errors,fitnesses) ordered in ascending 
  order of fitness
  """
  rankedWeights = [ item[0] for item in rankedPop]
  fitnessScores = [ item[-1] for item in rankedPop]
  newpopW = list(rankedWeights[:int(pop_size*0.15)])
  while len(newpopW) <= pop_size:                                       # Breed two randomly selected but different chromos until pop_size reached
    ch1, ch2 = [], []
    index1 = roulette(fitnessScores)                                    
    index2 = roulette(fitnessScores)
    while index1 == index2:                                             # ensures different chromos are used for breeeding 
      index2 = roulette(fitnessScores)
    #index1, index2 = 3,4
    ch1.extend(list(rankedWeights[index1]))
    ch2.extend(list(rankedWeights[index2]))
    if random.random() < crossover_rate: 
      ch1, ch2 = crossover(ch1, ch2)
    mutate(ch1)
    mutate(ch2)
    newpopW.append(ch1)
    newpopW.append(ch2)
  tst.dezip(rankedPop, rankedWeights, fitnessScores)
  tst.elitism(rankedWeights, newpopW)
  return newpopW