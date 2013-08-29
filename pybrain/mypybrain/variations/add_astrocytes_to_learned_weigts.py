# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:48:37 2013

The activation function was the hyperbolic tangent in all the
layers, except in the output layer where the threshold function was
used with a threshold value of 0.5 and an expected binary output.

All layers have astrocytes

Weight limiting? Decision of neurons will never change

Modification of weights - input or output weights? Potential implied 
method:
  * Input astrocytes modify output weights
  * Hidden astrocytes modify both input and output weights
  * Output astrocytes modify input weights

associated neuronal connections were active for at least n out of m
iterations (n: 2 to 3; m: 4, 6, 8), and 2) considering the time unit as
a single iteration, astrocytic effects lasted 4 to 8 iterations, and the
neuronal connection weights gradually increased (25%) or
decreased (50%) if the associated astrocyte was active or inactive,
respectively. 

 The combinations (Astrocytic Sensitivity, Neuron-glia power
connection: 2,4; 3,6; 2,6 y 3,8) were determined by trial-and-error,
and allowed an upper limit of 3, 4, 5 or 6 astrocytic activations,
respectively.

@author: david
"""
import random
import sys
sys.path.append('/home/david/Dropbox/programming/python/ann/mypybrain')

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import LinearLayer, TanhLayer
from pybrain.structure import FullConnection
from pybrain.structure import FeedForwardNetwork
from pybrain.tools.shortcuts import buildNetwork
from numpy import array, ones, zeros, append
import matplotlib.pyplot as plt
import numpy as np

from astrocyte_layer import AstrocyteLayer


def createDS():
    # taken from iris data set at machine learning repository
    pat  = [[[5.1, 3.5, 1.4, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[4.9, 3.0, 1.4, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[4.7, 3.2, 1.3, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[4.6, 3.1, 1.5, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.0, 3.6, 1.4, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.4, 3.9, 1.7, 0.4], [1, 0, 0], [0], ['Iris-setosa']], [[4.6, 3.4, 1.4, 0.3], [1, 0, 0], [0], ['Iris-setosa']], [[5.0, 3.4, 1.5, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[4.4, 2.9, 1.4, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[4.9, 3.1, 1.5, 0.1], [1, 0, 0], [0], ['Iris-setosa']], [[5.4, 3.7, 1.5, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[4.8, 3.4, 1.6, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[4.8, 3.0, 1.4, 0.1], [1, 0, 0], [0], ['Iris-setosa']], [[4.3, 3.0, 1.1, 0.1], [1, 0, 0], [0], ['Iris-setosa']], [[5.8, 4.0, 1.2, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.7, 4.4, 1.5, 0.4], [1, 0, 0], [0], ['Iris-setosa']], [[5.4, 3.9, 1.3, 0.4], [1, 0, 0], [0], ['Iris-setosa']], [[5.1, 3.5, 1.4, 0.3], [1, 0, 0], [0], ['Iris-setosa']], [[5.7, 3.8, 1.7, 0.3], [1, 0, 0], [0], ['Iris-setosa']], [[5.1, 3.8, 1.5, 0.3], [1, 0, 0], [0], ['Iris-setosa']], [[5.4, 3.4, 1.7, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.1, 3.7, 1.5, 0.4], [1, 0, 0], [0], ['Iris-setosa']], [[4.6, 3.6, 1.0, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.1, 3.3, 1.7, 0.5], [1, 0, 0], [0], ['Iris-setosa']], [[4.8, 3.4, 1.9, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.0, 3.0, 1.6, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.0, 3.4, 1.6, 0.4], [1, 0, 0], [0], ['Iris-setosa']], [[5.2, 3.5, 1.5, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.2, 3.4, 1.4, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[4.7, 3.2, 1.6, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[4.8, 3.1, 1.6, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.4, 3.4, 1.5, 0.4], [1, 0, 0], [0], ['Iris-setosa']], [[5.2, 4.1, 1.5, 0.1], [1, 0, 0], [0], ['Iris-setosa']], [[5.5, 4.2, 1.4, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[4.9, 3.1, 1.5, 0.1], [1, 0, 0], [0], ['Iris-setosa']], [[5.0, 3.2, 1.2, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.5, 3.5, 1.3, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[4.9, 3.1, 1.5, 0.1], [1, 0, 0], [0], ['Iris-setosa']], [[4.4, 3.0, 1.3, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.1, 3.4, 1.5, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.0, 3.5, 1.3, 0.3], [1, 0, 0], [0], ['Iris-setosa']], [[4.5, 2.3, 1.3, 0.3], [1, 0, 0], [0], ['Iris-setosa']], [[4.4, 3.2, 1.3, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.0, 3.5, 1.6, 0.6], [1, 0, 0], [0], ['Iris-setosa']], [[5.1, 3.8, 1.9, 0.4], [1, 0, 0], [0], ['Iris-setosa']], [[4.8, 3.0, 1.4, 0.3], [1, 0, 0], [0], ['Iris-setosa']], [[5.1, 3.8, 1.6, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[4.6, 3.2, 1.4, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.3, 3.7, 1.5, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[5.0, 3.3, 1.4, 0.2], [1, 0, 0], [0], ['Iris-setosa']], [[7.0, 3.2, 4.7, 1.4], [0, 1, 0], [1], ['Iris-versicolor']], [[6.4, 3.2, 4.5, 1.5], [0, 1, 0], [1], ['Iris-versicolor']], [[6.9, 3.1, 4.9, 1.5], [0, 1, 0], [1], ['Iris-versicolor']], [[5.5, 2.3, 4.0, 1.3], [0, 1, 0], [1], ['Iris-versicolor']], [[6.5, 2.8, 4.6, 1.5], [0, 1, 0], [1], ['Iris-versicolor']], [[5.7, 2.8, 4.5, 1.3], [0, 1, 0], [1], ['Iris-versicolor']], [[6.3, 3.3, 4.7, 1.6], [0, 1, 0], [1], ['Iris-versicolor']], [[4.9, 2.4, 3.3, 1.0], [0, 1, 0], [1], ['Iris-versicolor']], [[6.6, 2.9, 4.6, 1.3], [0, 1, 0], [1], ['Iris-versicolor']], [[5.2, 2.7, 3.9, 1.4], [0, 1, 0], [1], ['Iris-versicolor']], [[5.0, 2.0, 3.5, 1.0], [0, 1, 0], [1], ['Iris-versicolor']], [[5.9, 3.0, 4.2, 1.5], [0, 1, 0], [1], ['Iris-versicolor']], [[6.0, 2.2, 4.0, 1.0], [0, 1, 0], [1], ['Iris-versicolor']], [[6.1, 2.9, 4.7, 1.4], [0, 1, 0], [1], ['Iris-versicolor']], [[5.6, 2.9, 3.6, 1.3], [0, 1, 0], [1], ['Iris-versicolor']], [[6.7, 3.1, 4.4, 1.4], [0, 1, 0], [1], ['Iris-versicolor']], [[5.6, 3.0, 4.5, 1.5], [0, 1, 0], [1], ['Iris-versicolor']], [[5.8, 2.7, 4.1, 1.0], [0, 1, 0], [1], ['Iris-versicolor']], [[6.2, 2.2, 4.5, 1.5], [0, 1, 0], [1], ['Iris-versicolor']], [[5.6, 2.5, 3.9, 1.1], [0, 1, 0], [1], ['Iris-versicolor']], [[5.9, 3.2, 4.8, 1.8], [0, 1, 0], [1], ['Iris-versicolor']], [[6.1, 2.8, 4.0, 1.3], [0, 1, 0], [1], ['Iris-versicolor']], [[6.3, 2.5, 4.9, 1.5], [0, 1, 0], [1], ['Iris-versicolor']], [[6.1, 2.8, 4.7, 1.2], [0, 1, 0], [1], ['Iris-versicolor']], [[6.4, 2.9, 4.3, 1.3], [0, 1, 0], [1], ['Iris-versicolor']], [[6.6, 3.0, 4.4, 1.4], [0, 1, 0], [1], ['Iris-versicolor']], [[6.8, 2.8, 4.8, 1.4], [0, 1, 0], [1], ['Iris-versicolor']], [[6.7, 3.0, 5.0, 1.7], [0, 1, 0], [1], ['Iris-versicolor']], [[6.0, 2.9, 4.5, 1.5], [0, 1, 0], [1], ['Iris-versicolor']], [[5.7, 2.6, 3.5, 1.0], [0, 1, 0], [1], ['Iris-versicolor']], [[5.5, 2.4, 3.8, 1.1], [0, 1, 0], [1], ['Iris-versicolor']], [[5.5, 2.4, 3.7, 1.0], [0, 1, 0], [1], ['Iris-versicolor']], [[5.8, 2.7, 3.9, 1.2], [0, 1, 0], [1], ['Iris-versicolor']], [[6.0, 2.7, 5.1, 1.6], [0, 1, 0], [1], ['Iris-versicolor']], [[5.4, 3.0, 4.5, 1.5], [0, 1, 0], [1], ['Iris-versicolor']], [[6.0, 3.4, 4.5, 1.6], [0, 1, 0], [1], ['Iris-versicolor']], [[6.7, 3.1, 4.7, 1.5], [0, 1, 0], [1], ['Iris-versicolor']], [[6.3, 2.3, 4.4, 1.3], [0, 1, 0], [1], ['Iris-versicolor']], [[5.6, 3.0, 4.1, 1.3], [0, 1, 0], [1], ['Iris-versicolor']], [[5.5, 2.5, 4.0, 1.3], [0, 1, 0], [1], ['Iris-versicolor']], [[5.5, 2.6, 4.4, 1.2], [0, 1, 0], [1], ['Iris-versicolor']], [[6.1, 3.0, 4.6, 1.4], [0, 1, 0], [1], ['Iris-versicolor']], [[5.8, 2.6, 4.0, 1.2], [0, 1, 0], [1], ['Iris-versicolor']], [[5.0, 2.3, 3.3, 1.0], [0, 1, 0], [1], ['Iris-versicolor']], [[5.6, 2.7, 4.2, 1.3], [0, 1, 0], [1], ['Iris-versicolor']], [[5.7, 3.0, 4.2, 1.2], [0, 1, 0], [1], ['Iris-versicolor']], [[5.7, 2.9, 4.2, 1.3], [0, 1, 0], [1], ['Iris-versicolor']], [[6.2, 2.9, 4.3, 1.3], [0, 1, 0], [1], ['Iris-versicolor']], [[5.1, 2.5, 3.0, 1.1], [0, 1, 0], [1], ['Iris-versicolor']], [[5.7, 2.8, 4.1, 1.3], [0, 1, 0], [1], ['Iris-versicolor']], [[6.3, 3.3, 6.0, 2.5], [0, 0, 1], [2], ['Iris-virginica']], [[5.8, 2.7, 5.1, 1.9], [0, 0, 1], [2], ['Iris-virginica']], [[7.1, 3.0, 5.9, 2.1], [0, 0, 1], [2], ['Iris-virginica']], [[6.3, 2.9, 5.6, 1.8], [0, 0, 1], [2], ['Iris-virginica']], [[6.5, 3.0, 5.8, 2.2], [0, 0, 1], [2], ['Iris-virginica']], [[7.6, 3.0, 6.6, 2.1], [0, 0, 1], [2], ['Iris-virginica']], [[4.9, 2.5, 4.5, 1.7], [0, 0, 1], [2], ['Iris-virginica']], [[7.3, 2.9, 6.3, 1.8], [0, 0, 1], [2], ['Iris-virginica']], [[6.7, 2.5, 5.8, 1.8], [0, 0, 1], [2], ['Iris-virginica']], [[7.2, 3.6, 6.1, 2.5], [0, 0, 1], [2], ['Iris-virginica']], [[6.5, 3.2, 5.1, 2.0], [0, 0, 1], [2], ['Iris-virginica']], [[6.4, 2.7, 5.3, 1.9], [0, 0, 1], [2], ['Iris-virginica']], [[6.8, 3.0, 5.5, 2.1], [0, 0, 1], [2], ['Iris-virginica']], [[5.7, 2.5, 5.0, 2.0], [0, 0, 1], [2], ['Iris-virginica']], [[5.8, 2.8, 5.1, 2.4], [0, 0, 1], [2], ['Iris-virginica']], [[6.4, 3.2, 5.3, 2.3], [0, 0, 1], [2], ['Iris-virginica']], [[6.5, 3.0, 5.5, 1.8], [0, 0, 1], [2], ['Iris-virginica']], [[7.7, 3.8, 6.7, 2.2], [0, 0, 1], [2], ['Iris-virginica']], [[7.7, 2.6, 6.9, 2.3], [0, 0, 1], [2], ['Iris-virginica']], [[6.0, 2.2, 5.0, 1.5], [0, 0, 1], [2], ['Iris-virginica']], [[6.9, 3.2, 5.7, 2.3], [0, 0, 1], [2], ['Iris-virginica']], [[5.6, 2.8, 4.9, 2.0], [0, 0, 1], [2], ['Iris-virginica']], [[7.7, 2.8, 6.7, 2.0], [0, 0, 1], [2], ['Iris-virginica']], [[6.3, 2.7, 4.9, 1.8], [0, 0, 1], [2], ['Iris-virginica']], [[6.7, 3.3, 5.7, 2.1], [0, 0, 1], [2], ['Iris-virginica']], [[7.2, 3.2, 6.0, 1.8], [0, 0, 1], [2], ['Iris-virginica']], [[6.2, 2.8, 4.8, 1.8], [0, 0, 1], [2], ['Iris-virginica']], [[6.1, 3.0, 4.9, 1.8], [0, 0, 1], [2], ['Iris-virginica']], [[6.4, 2.8, 5.6, 2.1], [0, 0, 1], [2], ['Iris-virginica']], [[7.2, 3.0, 5.8, 1.6], [0, 0, 1], [2], ['Iris-virginica']], [[7.4, 2.8, 6.1, 1.9], [0, 0, 1], [2], ['Iris-virginica']], [[7.9, 3.8, 6.4, 2.0], [0, 0, 1], [2], ['Iris-virginica']], [[6.4, 2.8, 5.6, 2.2], [0, 0, 1], [2], ['Iris-virginica']], [[6.3, 2.8, 5.1, 1.5], [0, 0, 1], [2], ['Iris-virginica']], [[6.1, 2.6, 5.6, 1.4], [0, 0, 1], [2], ['Iris-virginica']], [[7.7, 3.0, 6.1, 2.3], [0, 0, 1], [2], ['Iris-virginica']], [[6.3, 3.4, 5.6, 2.4], [0, 0, 1], [2], ['Iris-virginica']], [[6.4, 3.1, 5.5, 1.8], [0, 0, 1], [2], ['Iris-virginica']], [[6.0, 3.0, 4.8, 1.8], [0, 0, 1], [2], ['Iris-virginica']], [[6.9, 3.1, 5.4, 2.1], [0, 0, 1], [2], ['Iris-virginica']], [[6.7, 3.1, 5.6, 2.4], [0, 0, 1], [2], ['Iris-virginica']], [[6.9, 3.1, 5.1, 2.3], [0, 0, 1], [2], ['Iris-virginica']], [[5.8, 2.7, 5.1, 1.9], [0, 0, 1], [2], ['Iris-virginica']], [[6.8, 3.2, 5.9, 2.3], [0, 0, 1], [2], ['Iris-virginica']], [[6.7, 3.3, 5.7, 2.5], [0, 0, 1], [2], ['Iris-virginica']], [[6.7, 3.0, 5.2, 2.3], [0, 0, 1], [2], ['Iris-virginica']], [[6.3, 2.5, 5.0, 1.9], [0, 0, 1], [2], ['Iris-virginica']], [[6.5, 3.0, 5.2, 2.0], [0, 0, 1], [2], ['Iris-virginica']], [[6.2, 3.4, 5.4, 2.3], [0, 0, 1], [2], ['Iris-virginica']], [[5.9, 3.0, 5.1, 1.8], [0, 0, 1], [2], ['Iris-virginica']]]
    alldata = ClassificationDataSet(4, 1, nb_classes=3, \
            class_labels=['set','vers','virg'])
    for p in pat:
        t = p[2]
        alldata.addSample(p[0],t)
    tstdata, trndata = alldata.splitWithProportion( 0.33 )
    trndata._convertToOneOfMany( )
    tstdata._convertToOneOfMany( )
    return trndata, tstdata


def roundActivationUpOrDown(x):
    assert x<=1 and x>=-1
    if x > 0:
        return 1
    else:
        return -1


"""
Although output layer should be binary with threshold bias layer of 0.5
and input layer should be tanh
"""
def createNN(indim, hiddim, outdim):
    nn = buildNetwork(indim, hiddim, outdim,
                    bias=False,
                    hiddenclass=TanhLayer, 
                    outclass=TanhLayer)
    nn.sortModules()
    return nn


def associateAstrocyteLayers(nn):
    in_to_hidden, = nn.connections[nn['in']]
    hidden_to_out, = nn.connections[nn['hidden0']]
    hiddenAstrocyteLayer = AstrocyteLayer(nn['hidden0'], in_to_hidden)
    outputAstrocyteLayer = AstrocyteLayer(nn['out'], hidden_to_out)
    return hiddenAstrocyteLayer, outputAstrocyteLayer


def plotErrorBar(data):
    means = np.mean(data, axis=0)
    std_errors = np.std(data, axis=0)/np.sqrt(len(data))
    plt.errorbar(range(len(means)), means, yerr=std_errors)
    plt.ylim(0,100)


def plotNoErrorBar(data, label='', xytxt=(0.8,1.02)):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)/np.sqrt(len(data))
    plt.plot(range(len(means)), means)
    plt.ylim(0,100)
    txt = label+'$\mu$='+str(means[-1])+'\n$\sigma$='+str(stds[-1])
    plt.annotate(txt, xy=(0.5,0.5), xytext=xytxt, xycoords='axes fraction',
           textcoords='axes fraction', fontsize='11')


repeats = 5
iterations = 3000

all_trn_results = []
all_tst_results = []

def main():
    trndata, tstdata = createDS()
    for repeat in xrange(repeats):
        iter_trn_results = []
        iter_tst_results = []
        nn = createNN(4, 6, 3)
        nn.randomize()
        hiddenAstrocyteLayer, outputAstrocyteLayer = associateAstrocyteLayers(nn)
        trainer = BackpropTrainer(nn, dataset=trndata, learningrate=0.01,
                                  momentum=0.1, verbose=False, weightdecay=0.0)
        for grand_iter in xrange(iterations):
            if grand_iter ==0:
                trainer.train()
            
            if grand_iter > 1000 and grand_iter < 2000:
                inputs  = list(trndata['input'])
                random.shuffle(inputs)
                for inpt in trndata['input']:
                    nn.activate(inpt)
                    for minor_iter in range(hiddenAstrocyteLayer.astrocyte_processing_iters):
                        hiddenAstrocyteLayer.update()
                        outputAstrocyteLayer.update()
                    hiddenAstrocyteLayer.reset()
                    outputAstrocyteLayer.reset()
    
    
            trainer.train()
          
            trnresult = percentError(trainer.testOnClassData(), trndata['class'])
            iter_trn_results.append(trnresult)
            tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
            iter_tst_results.append(tstresult)
    
            if not grand_iter%100:
                print 'epoch %4d' %trainer.totalepochs, 'train error %5.2f%%' %trnresult, \
                    'test error %5.2f%%' %tstresult
          
          
            all_trn_results.append(iter_trn_results)
            all_tst_results.append(iter_tst_results)

    path = '/home/david/Dropbox/programming/python/ann/mypybrain/'
    f = plt.figure(figsize=(10,5))
    plotErrorBar(all_trn_results)
    plotErrorBar(all_tst_results)
    plt.legend(('Training', 'Test'))
    f.savefig(path+'angn_bpThenAstro_error_bar.svg', format='svg')
    f = plt.figure(figsize=(10,5))
    side_text = 'Trials='+str(repeats)+'\n\nFinal\ntrain\nerror\n'
    plotNoErrorBar(all_trn_results, label=side_text, xytxt=(1.02, 0.7))
    plotNoErrorBar(all_tst_results, label='Final\ntest\nerror\n', xytxt=(1.02, 0.4))
    plt.legend(('Training', 'Test'))
    f.savefig(path+'angn_bpThenAstro_no_error_bar.svg', format='svg')
    plt.close()

if __name__ == '__main__':
  main()
