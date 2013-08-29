# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:19:42 2013

Standard backpropagation of iris data same architecture as angn

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
from datasets.iris import makeIrisDatasets
from plotting.plotters import *

def buildNN(indim=4, hiddim=6, outdim=3):
    net = FeedForwardNetwork()
    net.addInputModule(TanhLayer(indim, name = 'i'))
    net.addModule(TanhLayer(hiddim, name = 'h'))
    net.addOutputModule(ThresholdLayer(outdim, name = 'o', threshold=0.5))
    net.addConnection(FullConnection(net['i'], net['h']))
    net.addConnection(FullConnection(net['h'], net['o']))
    net.sortModules()
    return net

def trainNN(iterations, ds):
    errors = []
    nn = buildNN(4, 6, 3)
    nn.randomize()
    trainer = BackpropTrainer(nn, dataset=ds, learningrate=0.01,
                              momentum=0.1, verbose=False, weightdecay=0.0)
    for grand_iter in xrange(iterations):
        errors.append(trainer.train())
    return errors
    
def runExperiment():
    trials = range(5)
    iterations = 100
    errors = []
    for t in trials:
        print t
        training_ds, test_ds = makeIrisDatasets()
        errors.append(trainNN(iterations, training_ds))
    plt.subplot(3,1,1)
    plotMeansNoErrorBar(errors)
    plt.subplot(3,1,2)
    plotMeansErrorBar(errors)
    plt.subplot(3,1,3)
    plotAllTrials(errors)
    plt.show()
    
        
if __name__ == "__main__":
    runExperiment()