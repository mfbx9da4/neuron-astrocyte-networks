# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:01:07 2013

@author: david
"""

#learn digit classification with a nerual network

import pybrain
from pybrain.datasets import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.utilities           import percentError
import numpy

print "Importing training and test data"
data = numpy.genfromtxt('trainR.csv', delimiter = ',')
data = data[1:]
traindata = data[:(len(data)/2)]
testdata = data[(len(data)/2)+1:]

print "Importing actual data"
actualdata = numpy.genfromtxt('trainR.csv', delimiter = ',')

print "Adding samples to dataset and setting up neural network"
ds = ClassificationDataSet(784, 10, nb_classes = 10)
for x in traindata:
    ds.addSample(tuple(x[1:]),tuple(x[0:1]))
ds._convertToOneOfMany( bounds=[0,1] )
net = buildNetwork(784, 100, 10, bias=True, outclass=SoftmaxLayer)

print "Training the neural network"
trainer = BackpropTrainer(net, dataset=ds, momentum = 0.1,
                    verbose = True, weightdecay = 0.01)
for i in range(3):
    # train the network for 1 epoch
    trainer.trainEpochs( 1 )

    # evaluate the result on the training and test data
    trnresult = percentError( trainer.testOnClassData(), [x[0] for x in traindata] )

    # print the result
    print "epoch: " + str(trainer.totalepochs) + "  train error: " + str(trnresult)

print ""
print "Predicting with the neural network"
answerlist = []
for row in testdata:
    answer = numpy.argmax(net.activate(row[1:]))
    answerlist.append(answer)
tstresult = percentError(answerlist, [x[0] for x in testdata])
print "Test error: " + str(tstresult)