import unittest
import sys
import os
if os.environ['OS'] == 'Windows_NT':
    sys.path.append('C:\\Users\\david\\Dropbox\\programming\\python\\ann\\myangn\\sem6')
    sys.path.append('C:\\Users\\david\\Dropbox\\programming\\python\\ann\\mypybrain')
else:
    sys.path.append('/home/david/Dropbox/programming/python/ann/myangn/sem6')	

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.structure.connections import FullConnection
from pybrain.structure.networks.feedforward import FeedForwardNetwork
import numpy.testing as npt
from scipy import argmax

from mymodules.threshold import ThresholdLayer
import main
from iris import pybrainData, randomIrisData
from GA import *
from neuralnetwork import NN
from utilities import OutToClass


def createNN():
	nn = FeedForwardNetwork()
	inLayer = TanhLayer(4, name='in')
	hiddenLayer = TanhLayer(6, name='hidden0')
	outLayer = ThresholdLayer(3)
	nn.addInputModule(inLayer)
	nn.addModule(hiddenLayer)
	nn.addOutputModule(outLayer)
	in_to_hidden = FullConnection(inLayer, hiddenLayer)
	hidden_to_out = FullConnection(hiddenLayer, outLayer)
	nn.addConnection(in_to_hidden)
	nn.addConnection(hidden_to_out)
	nn.sortModules()
	return nn


class testLearnedWeights(unittest.TestCase):
	def setUp(self):
		# self.net = buildNetwork(4, 6, 3, bias=False, inclass=TanhLayer,
		# 	hiddenclass=TanhLayer, outclass=LinearLayer)
		# self.net.sortModules()
		self.net = createNN()
		self.trn_d, self.tst_d = pybrainData(0.01)
		self.trainer = BackpropTrainer(self.net, dataset=self.trn_d, 
			learningrate=0.01, momentum=0.1, verbose=True, weightdecay=0.0)
		self.trainer.trainEpochs(1)

	def testBPHasLearned(self):
		trnresult = percentError(self.trainer.testOnClassData(),
			self.trn_d['class'])
		tstresult = percentError(self.trainer.testOnClassData(dataset=self.tst_d),
			self.tst_d['class'])
		print 'trn perc error', trnresult
		print 'tst perc error', tstresult

	def testBPWeightsOnMyNetwork(self):
		pyb_ws = self.net.params.copy()
		pop = createPop()
		for nn in pop:
			nn.wi = pyb_ws[:nn.wi.size].reshape(NN.nh, NN.ni).T
			nn.wo = pyb_ws[nn.wi.size:].reshape(NN.no, NN.nh).T
		pairPop(pop, verbose=20)

	def testWeightsAndActivationsEquivalent(self):
		pyb_ws = self.net.params
		nn = NN()
		nn.wi = pyb_ws[:nn.wi.size].reshape(NN.nh, NN.ni).T
		nn.wo = pyb_ws[nn.wi.size:].reshape(NN.no, NN.nh).T
		for i, x in enumerate(self.trn_d['input']):
			nn.activate(x)
			out = self.net.activate(x)
			npt.assert_array_equal(nn.ai, self.net['in'].outputbuffer[0])
			# self.assertItemsEqual(list(nn.ah), list(self.net['hidden0'].outputbuffer[0]))
			for j, pb_ah in enumerate(self.net['hidden0'].outputbuffer[0]):
				self.assertAlmostEqual(nn.ah[j], pb_ah)
			for k, pb_ao in enumerate(out):
				self.assertAlmostEqual(nn.ao[k], pb_ao)

	def testDataAssignedCorrectly(self):
		NN.pat = zip(self.trn_d['input'], self.trn_d['target'])		
		pyb_ws = self.net.params.copy()
		nn = NN()
		nn.wi = pyb_ws[:nn.wi.size].reshape(NN.nh, NN.ni).T
		nn.wo = pyb_ws[nn.wi.size:].reshape(NN.no, NN.nh).T
		correct = 0
		wrong = 0
		all_aos = []
		for i, x in enumerate(self.trn_d['input']):
			nn.activate(x)
			out = self.net.activate(x)
			all_aos.append(nn.ao)
			if not (out - self.trn_d['target'][i]).any():
				correct += 1
			else:
				wrong += 1
		for i in range(len(array(NN.pat)[:,0])):
			npt.assert_array_equal(self.trn_d['input'][i], array(NN.pat)[:,0][i])
			npt.assert_array_equal(self.trn_d['input'][i], array(nn.pat)[:,0][i])
			npt.assert_array_equal(self.trn_d['target'][i], array(NN.pat)[:,1][i])
			npt.assert_array_equal(self.trn_d['target'][i], array(nn.pat)[:,1][i])
	
	def testPercentErrorIsSame(self):
		NN.pat = zip(self.trn_d['input'], self.trn_d['target'])		
		pyb_ws = self.net.params.copy()
		nn = NN()
		nn.wi = pyb_ws[:nn.wi.size].reshape(NN.nh, NN.ni).T
		nn.wo = pyb_ws[nn.wi.size:].reshape(NN.no, NN.nh).T
		correct = 0
		wrong = 0
		argmax_cor = 0
		argmax_wng = 0
		all_aos = []
		for i, x in enumerate(self.trn_d['input']):
			nn.activate(x)
			out = self.net.activate(x)
			# print 'ga bp trg', nn.ao, out, self.trn_d['target'][i], '++++' if not (out - self.trn_d['target'][i]).any() else '-'
			all_aos.append(nn.ao.copy())
			if not (out - self.trn_d['target'][i]).any():
				correct += 1
			else:
				wrong += 1
			if argmax(out) == argmax(self.trn_d['target'][i]):
				argmax_cor += 1
			else:
				argmax_wng += 1
		print 'actual', wrong, 'wrong', correct, 'correct', float(wrong) / (wrong + correct) * 100
		print 'using argmax', argmax_wng, 'wrong', argmax_cor, 'correct', float(argmax_wng) / (argmax_wng + argmax_cor) * 100
		argmax_perc_err = float(argmax_wng) / (argmax_wng + argmax_cor) * 100
		res = nn.sumErrors()
		nn_perc_err = 100 - res[1]
		pb_nn_perc_err = percentError(self.trainer.testOnClassData(), self.trn_d['class'])
		self.assertAlmostEqual(nn_perc_err, pb_nn_perc_err)
		self.assertAlmostEqual(nn_perc_err, pb_nn_perc_err, argmax_perc_err)

	# def testPrint(self):
	# 	NN.pat = zip(self.trn_d['input'], self.trn_d['target'])		
	# 	pyb_ws = self.net.params.copy()
	# 	nn = NN()
	# 	nn.wi = pyb_ws[:nn.wi.size].reshape(NN.nh, NN.ni).T
	# 	nn.wo = pyb_ws[nn.wi.size:].reshape(NN.no, NN.nh).T
	# 	correct = 0
	# 	wrong = 0
	# 	all_aos = []
	# 	for i, x in enumerate(self.trn_d['input']):
	# 		nn.activate(x)
	# 		out = self.net.activate(x)
	# 		print 'ga bp trg', nn.ao, out, self.trn_d['target'][i], '++++' if not (out - self.trn_d['target'][i]).any() else '-'
	# 		all_aos.append(nn.ao.copy())
	# 		if not (out - self.trn_d['target'][i]).any():
	# 			correct += 1
	# 		else:
	# 			wrong += 1
	# 	print wrong, 'wrong', correct, 'correct', float(wrong) / (wrong + correct) * 100
	# 	res = nn.sumErrors()
	# 	print 100 - res[1]
	# 	print percentError(self.trainer.testOnClassData(), self.trn_d['class'])
	# 	for i in range(len(array(NN.pat)[:,0])):
	# 		npt.assert_array_equal(self.trn_d['input'][i], array(NN.pat)[:,0][i])
	# 		npt.assert_array_equal(self.trn_d['input'][i], array(nn.pat)[:,0][i])
	# 		npt.assert_array_equal(self.trn_d['input'][i], array(res[3])[:,0][i])
	# 		npt.assert_array_equal(self.trn_d['target'][i], array(NN.pat)[:,1][i])
	# 		npt.assert_array_equal(self.trn_d['target'][i], array(nn.pat)[:,1][i])
	# 		npt.assert_array_equal(self.trn_d['target'][i], array(res[3])[:,1][i])
	# 	assert array(res[2]).shape == array(all_aos).shape
	# 	print array(res[2]).shape, array(all_aos).shape
	# 	for x, y in zip(res[2], all_aos):
	# 		npt.assert_array_equal(x, y)
		
	# 	outs, trgs = OutToClass(res[2], array(NN.pat)[:,1])
	# 	pb_outs, pb_trgs = self.trainer.testOnClassData(), self.trn_d['class']
	# 	for w, x, y, z in zip(outs, trgs, pb_outs, pb_trgs):
	# 		npt.assert_array_equal(w, y)
	# 		npt.assert_array_equal(x, z)
		





if __name__ == '__main__':
	unittest.main()