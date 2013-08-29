#-*- encoding: utf8 -*- 

# A python script to test and understand pybrain basics
# Based in Martin Felder's FNN pybrain example

from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer,RPropMinusTrainer
from pybrain.utilities           import percentError

from generatedata import *


training_dataset = generate_data()
# neural networks work better if classes are encoded using 
# one output neuron per class
training_dataset._convertToOneOfMany( bounds=[0,1] )

# same for the independent test data set
testing_dataset = generate_data(test=True)
testing_dataset._convertToOneOfMany( bounds=[0,1] )

# build a feed-forward network with 20 hidden units, plus 
# a corresponding trainer
fnn = buildNetwork( training_dataset.indim, 15,15, training_dataset.outdim, outclass=SoftmaxLayer )
#trainer = BackpropTrainer( fnn, dataset=training_dataset,verbose=True)
trainer = RPropMinusTrainer( fnn, dataset=training_dataset, verbose=True )

for i in range(500):
    # train the network for 1 epoch
    trainer.trainEpochs( 15 )
    
    # evaluate the result on the training and test data
    trnresult = percentError( trainer.testOnClassData(), 
                              training_dataset['class'] )
    tstresult = percentError( trainer.testOnClassData( 
           dataset=testing_dataset ), testing_dataset['class'] )

    # print the result
    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
    if tstresult <= 0.5 :
         print 'Bingo !!!!!!!!!!!!!!!!!!!!!!'
         break
