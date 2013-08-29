__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import randn, zeros, where, array
from random import choice, random, gauss
import math

from evolution import Evolution
from pybrain.optimization.optimizer import ContinuousOptimizer


class GA(ContinuousOptimizer, Evolution):
    """ Standard Genetic Algorithm. """

    #: selection scheme
    tournament = False
    tournamentSize = 2

    #: selection proportion
    topProportion = 0.2

    elitism = False
    eliteProportion = 0.5
    _eliteSize = None # override with an exact number

    #: mutation probability
    mutationProb = 0.1
    mutationStdDev = 0.5
    initRangeScaling = 10.

    initialPopulation = None

    mustMaximize = True

    def initPopulation(self):
        if self.initialPopulation is not None:
            self.currentpop = self.initialPopulation
        else:
            self.currentpop = [self._initEvaluable]
            for _ in range(self.populationSize-1):
                self.currentpop.append(self._initEvaluable+randn(self.numParameters)
                                       *self.mutationStdDev*self.initRangeScaling)

    def crossOver(self, parents, nbChildren):
        """ generate a number of children by doing 1-point cross-over
        main loop:
            * selects two parents
            * crossover to gen one child
            """
        xdim = self.numParameters
        # assert xdim == parents[0][0].shape[0]
        children = []
        diff = 0
        for i in range(nbChildren):
            if xdim < 2:
                children.append(choice(parents))
            else:
                res = zeros(xdim)
                point = choice(range(xdim-1))
                if not self.tournament:
                    p1 = choice(parents)
                    p2 = choice(parents)
                    c = (p1 - p2).all()
                    print p1.shape
                    diff += where(c, 1, 0)
                else:
                    p1, p2 = parents[i]
                    print 'p1', p1.shape
                    print 'p2', p2.shape
                    print self._allGenerations[0][0][0].shape
                res[:point] = p1[:point]
                res[point:] = p2[point:]
                children.append(res)
        assert diff < nbChildren
        print diff / float(nbChildren)
        print array(children).shape
        return children

    def mutated(self, indiv):
        """ mutate some genes of the given individual """
        res = indiv.copy()
        for i in range(self.numParameters):
            if random() < self.mutationProb:
                res[i] = indiv[i] + gauss(0, self.mutationStdDev)
        return res

    @property
    def selectionSize(self):
        """ the number of parents selected from the current population """
        return int(self.populationSize * self.topProportion)

    @property
    def eliteSize(self):
        if self.elitism:
            if self._eliteSize != None:
                return self._eliteSize
            else:
                return int(self.populationSize * self.eliteProportion)
        else:
            return 0

    def select(self):
        """ select some of the individuals of the population, taking into account their fitnesses

        :return: list of selected parents """
        if not self.tournament:
            tmp = zip(self.fitnesses, self.currentpop)
            tmp.sort(key = lambda x: x[0])
            tmp2 = list(reversed(tmp))[:self.selectionSize]
            return map(lambda x: x[1], tmp2)
        else:
            return self.tournamentSelect()

    def tournamentSelect(self):
        parents = []
        params, fitnesses = self._allGenerations[-1]
        fraction_fitnesses = calcFitsAsFraction(fitnesses)
        same = 0
        while len(parents) != (self.populationSize-self.eliteSize) * 2:
            ind1 = roulette(fraction_fitnesses)                                    
            ind2 = roulette(fraction_fitnesses)
            same += where(ind1 == ind2, 1, 0)
            p1 = params[ind1].copy() 
            p2 = params[ind2].copy()
            parents.append(array([p1, p2]))
        assert same != len(parents)
        print same / float(len(parents))
        return parents

    def produceOffspring(self):
        """ produce offspring by selection, mutation and crossover. 
            * select parents
            * conserve elite. if selection size is smaller conserve than 
            * for each child in the crossovered chromosomes
        """
        parents = self.select()
        es = min(self.eliteSize, self.selectionSize)
        self.currentpop = parents[:es]
        for child in self.crossOver(parents, self.populationSize-es):
            self.currentpop.append(self.mutated(child))



def roulette(fitness_scores):
  cumalative_fitness = 0.0
  r = random()
  for i in range(len(fitness_scores)): 
    cumalative_fitness += fitness_scores[i]
    if cumalative_fitness > r: 
      return i

def calcFitsAsFraction(numbers):  
    """each fitness is a fraction of the total error"""
    total, fitnesses = math.fabs(sum(numbers)), []
    for i in range(len(numbers)):           
        try:
              fitness = math.fabs(numbers[i]) / total
        except ZeroDivisionError:
              print 'individual outputted zero correct responses'
              fitness = 0
        fitnesses.append(fitness)
    assert sum(fitnesses) < 1.000001, sum(fitnesses)
    return fitnesses