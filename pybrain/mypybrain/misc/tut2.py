from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure import LinearLayer, SigmoidLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

# generates random data in two dimensions
# http://en.wikipedia.org/wiki/Multivariate_normal_distribution
means = [(-1,0),(2,4),(3,1)]
cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
alldata = ClassificationDataSet(2, 1, nb_classes=3)
for n in xrange(400):
    for klass in range(3):
        input = multivariate_normal(means[klass],cov[klass])
        alldata.addSample(input, [klass])

print alldata['input'][0], alldata['target'][0]


# splits data into 25%:75% respectively
tstdata, trndata = alldata.splitWithProportion( 0.25 )

# target was previously intergeer is now a pattern of output nodes and 
# class values are integers
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )


print "DATA:\nNumber of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

# shortcut for building network of 2,5,3 with output as softmaxlayer
# softmax layer ensures outputs are between 0-1 and sum to 1 which is required for multivariate
# probability distribution
fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )

# weight decay prevents stabilization of weights to 0/1 which prevents learning
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

# generates a square grid of data points and put it into a dataset
# the output values of these inputs will be used for the visualization
ticks = arange(-3.,6.,0.2)
# returns two coordinate matrices for two coordinate vectors
X, Y = meshgrid(ticks, ticks)
# need column vectors in dataset, not arrays, of same dimension as network
griddata = ClassificationDataSet(2,1, nb_classes=3)
for i in xrange(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0]) # target can be ignored
griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy

for i in range(20):
  trainer.trainEpochs(1)
  # percent error is mismatch between output and target
  trnresult = percentError(trainer.testOnClassData(), trndata['class'])
  tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
  print 'epoch %4d' %trainer.totalepochs, \
        'train error %5.2f%%' %trnresult, \
        'test error %5.2f%%' %tstresult
  out = fnn.activateOnDataset(griddata)
  out = out.argmax(axis=1) # col=0 row=1 returns index of largest in row
  out = out.reshape((X.shape))

  figure(1)
  ioff()
  clf()
  hold(True)
  for c in range(3):
    here, _ = where(tstdata['class']==c)
    plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
  if out.max()!=out.min():  # safety check against flat field
    contourf(X, Y, out)   # plot the contour
  ion()   # interactive graphics on
  draw()  # update the plot
f = meshgrid(range(4), range(4))

ioff()
show()  
  
  
