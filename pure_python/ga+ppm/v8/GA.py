from operator import itemgetter
import random
from random import shuffle
from copy import deepcopy

from pylab import array

from utilities import crossover, mutate, roulette, calcFit
from neuralnetwork import NN
from astrocyte_layer import AstrocyteLayer


def createPop():
    return [NN() for i in range(NN.pop_size)]


def pairPop (pop, verbose=True):
    """Takes a population of NNs and calculates the error and returns a
    list containing the weights of each individual with its associated 
    errors and fitnesses"""
    weights, inv_errs, perc_accs = [], [], []
    for i in range(len(pop)):    
        weights.append([pop[i].wi.copy(), pop[i].wo.copy()])
        inv_err, perc_acc = pop[i].sumErrors()
        inv_errs.append(inv_err)
        perc_accs.append(perc_acc)
    fitnesses = calcFit(perc_accs)
    if verbose:
        for i in range(int(NN.eliteN)): 
            print str(i).zfill(2), '1/sum(MSEs)', str(perc_accs[i]).rjust(15), \
                str(int(perc_accs[i] * NN.graphical_error_scale) * '-').rjust(20), \
                'fitness'.rjust(12), str(fitnesses[i]).rjust(17), \
                str(int(fitnesses[i] * 1000) * '-').rjust(20)
        print 'lowest fittness', min(fitnesses)
    return zip(weights, inv_errs, perc_accs, fitnesses)

  
def rankPop(newpopW, pop):
    errors, copy = [], []
    for i in range(NN.pop_size):  
        pop[i].assignWeights(newpopW[i])                             
    for i in range(NN.pop_size):  
        pop[i].testWeights(newpopW[i])
    pairedPop = pairPop(pop)
    rankedPop = sorted(pairedPop, key=itemgetter(-1), reverse=True) 
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
    ch1 = [x.copy() for x in rankedWeights[ind1]]
    ch2 = [x.copy() for x in rankedWeights[ind2]]
    return ch1, ch2


def trainPP(new_weights, pop, **kwargs):
    for i in range(NN.pop_size):
        pop[i].assignWeights(new_weights[i])
    inputs = array(NN.pat)[:,0]
    shuffle(inputs)
    for ind in pop[NN.eliteN:]:
        hidAstroL, outAstroL = associateAstrocytes(ind, **kwargs)
        for inp in inputs:
            ind.activate(inp)  
            for m in range(hidAstroL.astro_processing_iters):
                hidAstroL.update()
                outAstroL.update()
            hidAstroL.reset()
            outAstroL.reset()
    new_weights = []
    for i in range(len(pop)):    
        new_weights.append([pop[i].wi.copy(), pop[i].wo.copy()])
    return new_weights

def associateAstrocytes(net, **kwargs):
    hiddenAstrocyteLayer = AstrocyteLayer(net.ah, net.wi, **kwargs)
    outputAstrocyteLayer = AstrocyteLayer(net.ao, net.wo, **kwargs)
    return hiddenAstrocyteLayer, outputAstrocyteLayer









