# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 22:41:38 2013

Basic Backprop implementation of Iris flower problem

@author: david
"""

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer, TanhLayer
from pybrain.structure import LinearLayer, SigmoidLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from training_patterns import boolpat

alldata = ClassificationDataSet(4, 1, nb_classes=3, \
          class_labels=['set','vers','virg'])
for p in boolpat:
  t = [p[1][0]+1]
  alldata.addSample(p[0],t)

# splits data into 25%:75% respectively
tstdata, trndata = alldata.splitWithProportion( 0.33 )
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )



print "DATA:\nNumber of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0], \
      trndata.getClass(int(trndata['class'][0][0]))

# shortcut for building network of 2,5,3 with output as softmaxlayer
# softmax layer ensures outputs are between 0-1 and sum to 1 which is required for multivariate
# probability distribution
fnn = buildNetwork(trndata.indim, 6, trndata.outdim, hiddenclass=TanhLayer, \
                  outclass=TanhLayer)


# weight decay prevents stabilization of weights to 0/1 which prevents learning
trainer = BackpropTrainer(fnn, dataset=trndata, learningrate=0.2, \
          momentum=0.1, verbose=True, weightdecay=0.0)

for i in range(10):
  trainer.trainEpochs(1)
  # percent error is mismatch between output and target
  trnresult = percentError(trainer.testOnClassData(), trndata['class'])
  tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
  print 'epoch %4d' %trainer.totalepochs, \
        'train error %5.2f%%' %trnresult, \
        'test error %5.2f%%' %tstresult
