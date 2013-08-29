# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:00:46 2013

To prove that bp algorithm provided by pybrain learns sufficiently fast

@author: david
"""

import sys
sys.path.append('/home/david/Dropbox/programming/python/ann/mypybrain')

from pybrain.structure import LinearLayer, TanhLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FullConnection
import matplotlib.pyplot as plt

from astrocyte_layer import AstrocyteLayer
from mymodules.threshold import ThresholdLayer
from datasets.parity import ParityDataSet
from plotting.plotters import *
from plotting.results import Result


def buildNN(indim, hiddim, outdim):
    net = FeedForwardNetwork()
    net.addInputModule(TanhLayer(indim, name = 'i'))
    net.addModule(TanhLayer(hiddim, name = 'h'))
    net.addOutputModule(ThresholdLayer(outdim, name = 'o', threshold=0.5))
    net.addConnection(FullConnection(net['i'], net['h']))
    net.addConnection(FullConnection(net['h'], net['o']))
    net.sortModules()
    return net


def trainParityNN(iterations, ds):
    errors = ['errors']
    weights = ['weights']
    activations = ['activations']
    nn = buildNN(ds.indim, ds.indim+ds.indim/2, ds.outdim)
    nn.randomize()
    trainer = BackpropTrainer(nn, dataset=ds, learningrate=0.01,
                              momentum=0.1, verbose=False, weightdecay=0.0)
    for grand_iter in xrange(iterations):
        errors.append(trainer.train())
        weights.append([x for x in nn.params])
        temp_acts = [ array([x for x in mod.outputbuffer[:][0]]) for mod in nn.modules]
        activations.append(sorted(temp_acts, key=len))
#    print activations
    return [errors, weights, activations]
    
    
def runExperiment():
    trials = range(4)
    iterations = 10
    results = Result()
    for t in trials:
        print t
        ds = ParityDataSet(nbits=4, nsamples=50)
        results.append(trainParityNN(iterations, ds))
#    raw_input('')
    plt.subplot(3,1,1)
    plotErrorsMeansNoErrorBar(results.data)
    plt.subplot(3,1,2)
    plotErrorsMeansErrorBar(results.data)
    plt.subplot(3,1,3)
    plotErrorsAllTrials(results.data)
    plt.figure()
    plotWeightsAllTrials(results.data)
    plt.figure()
    plotActivationsAllTrials(results.data, (1, 4, 6))
    plt.show()
    
        
if __name__ == "__main__":
    runExperiment()