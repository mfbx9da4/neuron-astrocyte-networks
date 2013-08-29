# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:48:37 2013

@author: david
"""

from pybrain.structure.modules.neuronlayer import NeuronLayer

class QuadraticPolynomialLayer(NeuronLayer):

    def _forwardImplementation(self, inbuf, outbuf):
      outbuf[:] = inbuf*1
#      print 'here is the output of this layer', map(round, outbuf)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
      inerr[:] = 2 * inbuf * outerr

from pybrain.structure.connections.connection import Connection
from pybrain.structure.parametercontainer import ParameterContainer
class IdentityConnection(Connection, ParameterContainer):
     """Connection which connects the i'th element from the first module's output
     buffer to the i'th element of the second module's input buffer."""

     def __init__(self, *args, **kwargs):
         Connection.__init__(self, *args, **kwargs)
         ParameterContainer.__init__(self, paramdim=self.indim*self.outdim)

     def _forwardImplementation(self, inbuf, outbuf):
         # outbuff is neurons in hidden layer inbuff is input layer inputs
         # added as hidden neurons may receive inputs from other connections
         outbuf += dot(reshape(self.params, (self.outdim,self.indim)), inbuf)

     # for back prop     
     def _backwardImplementation(self, outerr, inerr, inbuf):
         inerr += dot(reshape(self.params, (self.outdim, self.indim)).T, outerr)
         ds = self.derivs         
         ds += outer(inbuf, outerr).T.flatten()

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer, TanhLayer
from pybrain.structure import LinearLayer, SigmoidLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where

alldata = ClassificationDataSet(4, 1, nb_classes=3, \
          class_labels=['set','vers','virg'])
for p in boolpat:
  t = p[2]
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
print len(trndata['target'])

from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import FeedForwardNetwork

n= FeedForwardNetwork()
inLayer = LinearLayer(4, name='i')
hiddenLayer = TanhLayer(4, name='h')
outLayer = TanhLayer(3, name='o')
n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)
n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)
n.sortModules()

n.activate(trndata['input'][0])
print 'i', n['i'].outputbuffer
print 'h', n['h'].outputbuffer
print 'o', n['o'].outputbuffer
  
# weight decay prevents stabilization of weights to 0/1 which prevents learning
trainer = BackpropTrainer(n, dataset=trndata, learningrate=0.2, \
          momentum=0.1, verbose=True, weightdecay=0.0)

trnresults, tstresults = [], []
for i in range(50):
  trainer.trainEpochs(1)
  # percent error is mismatch between output and target
  trnresult = percentError(trainer.testOnClassData(), trndata['class'])
  trnresults.append(trnresult)
  tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
  tstresults.append(tstresult)
  print 'epoch %4d' %trainer.totalepochs, \
        'train error %5.2f%%' %trnresult, \
        'test error %5.2f%%' %tstresult
