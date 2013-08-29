# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:22:38 2013

@author: david
"""

import random
import math

import matplotlib
from pylab import plot, legend, subplot, grid, xlabel, ylabel, show, title

from pyneurgen.neuralnet import NeuralNet
from pyneurgen.nodes import BiasNode, Connection


#   all samples are drawn from this population
pop_len = 200
factor = 1.0 / float(pop_len)
population = [[i, math.sin(float(i) * factor * 10.0) + \
                random.gauss(float(i) * factor, .2)]
                    for i in range(pop_len)]

all_inputs = []
all_targets = []

def population_gen(population):
    """
    This function shuffles the values of the population and yields the
    items in a random fashion.

    """

    pop_sort = [item for item in population]
    random.shuffle(pop_sort)

    for item in pop_sort:
        yield item
        
#   Build the inputs
for position, target in population_gen(population):
    pos = float(position)
    all_inputs.append([random.random(), pos * factor])
    all_targets.append([target])
    
    
    
    
    
    
    
    
    
    
    
    
    
    