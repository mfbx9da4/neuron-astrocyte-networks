    # -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:53:37 2013

@author: david
"""

import sys
import os

if os.environ['OS'] == 'Windows_NT':
    sys.path.append('C:\\Users\\david\\Dropbox\\' +
    'programming\\python\\ann\\myangn\\sem6')
    sys.path.append('C:\\Users\\david\\Dropbox\\' +
    'programming\\python\\ann\\mypybrain')
else:
    sys.path.append('/home/david/Dropbox/programming/python/ann/myangn/sem6')
    sys.path.append('/home/david/Dropbox/programming/python/ann/mypybrain')


from pylab import array, ylim, where, average
from pylab import plot, legend, subplot, grid, xlabel, ylabel, show, title
from pyneurgen.neuralnet import NeuralNet
from pyneurgen.nodes import BiasNode, Connection
from pybrain.utilities import percentError

from iris import neurgenData
from src.utilities import percentError

def buildSamples():
	all_inputs, all_targets = neurgenData()
	return all_inputs, all_targets

def buildIrisNetwork(all_inputs, all_targets):
	net = NeuralNet()
	net.init_layers(4, [6], 3)

	net.randomize_network()
	net.set_halt_on_extremes(True)

	#   Set to constrain beginning weights to -.5 to .5
	#       Just to show we can
	#net.set_random_constraint(.5)
	net.set_learnrate(.1)

	net.set_all_inputs(all_inputs)
	net.set_all_targets(all_targets)

	length = len(all_inputs)
	learn_end_point = int(length * .5)

	net.set_learn_range(0, learn_end_point)
	net.set_test_range(learn_end_point + 1, length-1)

	net.layers[0].set_activation_type('tanh')
	net.layers[1].set_activation_type('tanh')
	net.layers[2].set_activation_type('threshold')
	return net

all_inputs, all_targets = buildSamples()
net = buildIrisNetwork(all_inputs, all_targets)

iterations = 50
trn_errors = []
tst_errors = []
tst_perc_errors = []
################################
#	left off: was checking whether normal GA is learning same as
#	learn epochs=1. need to check percent error validation is all 
# 	correct. Write tests!!!!
############################
for i in range(iterations):
	print i,
	net.learn(epochs=1, show_epoch_results=True, 
			  random_testing=True)
# net.learn(epochs=500, show_epoch_results=True, 
# 			  random_testing=True)

	# outs = []
	# trgs = []
	# for inp, trg in net.get_test_data():
	# 	net.process_sample(inp, trg)
	# 	out = net.output_layer.activations()
	# 	vals = net.output_layer.values()
	# 	net.output_layer.get_errors()
	# 	outs.append(out)
	# 	trgs.append(trg)
	trn_errors.extend(net.accum_mse)
	tst_errors.append(net.test())
	trgs_activs = array(net.test_targets_activations)
	tst_perc_errors.append(percentError(trgs_activs[:,0], trgs_activs[:,1]))

targets = list(array(net.test_targets_activations)[:,0])
activations = list(array(net.test_targets_activations)[:,1])
all_targets1 = [ int(where(x == 1.0)[0]) for x in targets]
allactuals = [ where(x == 1.0)[0] for x in activations]
allactuals = [ x[0] if len(x) ==  1 else 3 for x in allactuals] 
print array(net.test_targets_activations).shape
print array(all_targets1).shape, max(all_targets1), min(all_targets1)
print array(allactuals).shape, max(allactuals), min(allactuals)
print percentError(all_targets1, allactuals)
print percentError(allactuals, all_targets1)
print array(trn_errors).shape
print array(tst_errors).shape
print array(tst_perc_errors).shape

subplot(3, 1, 1)
plot(all_targets1, 'bo', label='targets', markersize=10.0)
plot(allactuals, 'ro', label='actuals')
grid(True)
ylim(-0.2, 3.2)

#legend(loc='lower left', numpoints=1)
title("Test Target Points (blue) vs Actual Points (red)")

subplot(3, 1, 2)
plot(range(1, len(trn_errors) + 1, 1), trn_errors)
plot(range(1, len(trn_errors) + 1, 1), tst_errors)
xlabel('epochs')
ylabel('mean squared error')
grid(True)
title("Mean Squared Error by Epoch")

subplot(3, 1, 3)
plot(range(1, len(trn_errors) + 1, 1), tst_perc_errors)
xlabel('epochs')
ylabel('percent errors')
ylim(0, 100)
grid(True)
title("Percentage errors")

show()


