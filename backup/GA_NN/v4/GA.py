from operator import itemgetter
import random
from copy import deepcopy

from pylab import array

from utilities import crossover, mutate, roulette, calcFit
from neuralnetwork import NN


def createPop():
    return [NN() for i in range(NN.pop_size)]


def pairPop (pop, verbose=True):
    """Takes a population of NNs and calculates the error and returns a
    list containing the weights of each individual with its associated 
    errors and fitnesses"""
    weights, inv_errs, perc_errs = [], [], []
    for i in range(len(pop)):    
        weights.append([pop[i].wi.copy(), pop[i].wo.copy()])
        inv_err, perc_err = pop[i].sumErrors()
        inv_errs.append(inv_err)
        perc_errs.append(perc_err)
    fitnesses = calcFit(inv_errs)
    if verbose:
        for i in range(int(NN.pop_size * 0.15)): 
            print str(i).zfill(2), '1/sum(MSEs)', str(inv_errs[i]).rjust(15), \
                str(int(inv_errs[i] * NN.graphical_error_scale) * '-').rjust(20), \
                'fitness'.rjust(12), str(fitnesses[i]).rjust(17), \
                str(int(fitnesses[i] * 1000) * '-').rjust(20)
    return zip(weights, inv_errs, perc_errs, fitnesses)

  
def rankPop(newpopW, pop):
    errors, copy = [], []
    # don't think this is necessary:
    # pop = createPop()
    for i in range(NN.pop_size):  
        pop[i].assignWeights(newpopW[i])                                    
    # do i need to keep the below lines? may be unnecessary now
    for i in range(NN.pop_size):  
        pop[i].testWeights(newpopW[i])
    pairedPop = pairPop(pop)
    rankedPop = sorted(pairedPop, key=itemgetter(-1), reverse=True)   # weights are sorted in descending order of fitness (fittest first)
    errors = [x[1] for x in rankedPop]
    top_mse = 1 / errors[0]
    avg_mse = 1 / (sum(errors) / float(len(errors)))
    perc_errs = [x[2] for x in rankedPop]
    top_perc = perc_errs[0]
    avg_perc = sum(perc_errs) / float(len(perc_errs))
    return rankedPop, top_mse, avg_mse, top_perc, avg_perc


def evolveNewPop(rankedPop):
    """
    rankedPop is zip(weights, errors, fitnesses) ordered in ascending 
    order of fitness
    """
    # would be quicker if was array rankedPop[:,0]
    rankedWeights = [item[0] for item in rankedPop]
    fitnessScores = [item[-1] for item in rankedPop]
    # could try .wi.copy() for speed
    newpopW = deepcopy(rankedWeights[:NN.eliteN])
    while len(newpopW) < NN.pop_size:
        ch1, ch2 = selectTwoIndividuals(fitnessScores, rankedWeights)
        if random.random() <= NN.crossover_rate: 
          ch1, ch2 = crossover(ch1, ch2, NN)
        mutate(ch1, NN.mutation_rate)
        mutate(ch2, NN.mutation_rate)
        newpopW.append(ch1)
        newpopW.append(ch2)
    return newpopW[:NN.pop_size]


def selectTwoIndividuals(fitnessScores, rankedWeights):
    ind1 = roulette(fitnessScores)                                    
    ind2 = roulette(fitnessScores)
    # Variation: is this an unneccessary bottleneck?
    while ind1 == ind2:
      ind2 = roulette(fitnessScores)
    ch1 = deepcopy(rankedWeights[ind1])
    ch2 = deepcopy(rankedWeights[ind2])
    return ch1, ch2


