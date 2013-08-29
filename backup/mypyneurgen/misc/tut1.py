# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:53:37 2013

@author: david
"""

import random
import math
import sys
sys.path.append('/home/david/Dropbox/programming/python/ann/myangn/sem6')

import matplotlib
from pylab import array
from pylab import plot, legend, subplot, grid, xlabel, ylabel, show, title
from pyneurgen.neuralnet import NeuralNet
from pyneurgen.nodes import BiasNode, Connection

from iris import randomdata

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
pat, testpat = randomdata()
for p in pat:
    all_inputs.append(p[0])
    all_targets.append(p[1])
    
net = NeuralNet()
net.init_layers(4, [6], 3)

print array(all_inputs).shape
print all_inputs
print 
print 
print array(all_targets).shape
print all_targets

raw_input('')

net.randomize_network()
net.set_halt_on_extremes(True)

#   Set to constrain beginning weights to -.5 to .5
#       Just to show we can
net.set_random_constraint(.5)
net.set_learnrate(.1)

net.set_all_inputs(all_inputs)
net.set_all_targets(all_targets)

length = len(all_inputs)
learn_end_point = int(length * .8)

net.set_learn_range(0, learn_end_point)
net.set_test_range(learn_end_point + 1, length - 1)

net.layers[1].set_activation_type('tanh')

net.learn(epochs=125, show_epoch_results=True,
    random_testing=False)
    
mse = net.test()


test_positions = [item[0][1] * 1000.0 for item in net.get_test_data()]

all_targets1 = [item[0][0] for item in net.test_targets_activations]
allactuals = [item[1][0] for item in net.test_targets_activations]

#   This is quick and dirty, but it will show the results
subplot(3, 1, 1)
plot([i[1] for i in population])
title("Population")
grid(True)

subplot(3, 1, 2)
plot(test_positions, all_targets1, 'bo', label='targets')
plot(test_positions, allactuals, 'ro', label='actuals')
grid(True)
legend(loc='lower left', numpoints=1)
title("Test Target Points vs Actual Points")

subplot(3, 1, 3)
plot(range(1, len(net.accum_mse) + 1, 1), net.accum_mse)
xlabel('epochs')
ylabel('mean squared error')
grid(True)
title("Mean Squared Error by Epoch")

show()
    
    
    
    
    
    
    