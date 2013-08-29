# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:35:35 2013

results object to contain resutls data for all trials

@author: david
"""

from pylab import *

class Result():
    """Data object for storing output from neural networks"""
    def __init__(self):
        self.data = {'errors': [],
                     'percentErrors': [],
                     'weights': [],
                     'activations': []
                     }
    def append(self, d):
        """This is to be called at the end of each trial
        key:
            d: output results data from algorithm
            x: a parameter of d data 
        """
        for x in d:
            xname = x[0] 
            assert xname in self.data.keys()
            if xname == 'errors':
                self.data[xname].append(x[1:])
            elif xname == 'weights':
                """
                Given in the form:
                    iter1 = [array([nn.params]), array([nn.params]) ... ]
                    trial1 = ['weights', [iter1], [iter2] ... ]
                
                transpose converts into:
                    trial1 = [[weight1ValsForAllIters], [weight2ValsForAllIters], ... ]
                """
                weights = array([y for y in x[1:]]).T
                self.data[xname].append(weights)
            elif xname == 'activations':
                """
                Given in the form:
                    iter1 = [array([layer1Vals]), array([layer2Vals]) ... ]
                    trial1 = ['activations', [iter1], [iter2], ... ]
                    
                array(x[1:]).T converts trial1 into form:
                    trial1 = [[layer1ValsForAllIters], [layer2ValsForAllIters], ... ]
                    
                To convert into each neuron for all iters can't use [1] because
                layerVals inherits data type 'object' and therefore transpose 
                does not work. Therefore must use [y for y in layerVals]
                 
                 [1] for layerVals in array(x[1:]).T:
                         # use extend so that each item is a weight
                         activations.extend(layerVals.T)
                """
                for iterVals in x[1:]:
                    assert sorted(iterVals, key=len) == iterVals
                
                activations = []
                for layerVals in array(x[1:]).T:
                    activations.extend(array([y for y in layerVals]).T)
                self.data[xname].append(activations)
