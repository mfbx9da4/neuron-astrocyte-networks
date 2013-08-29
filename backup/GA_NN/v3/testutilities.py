from copy import deepcopy, copy
import unittest
import random
from operator import itemgetter

from pylab import array, where, linspace, ones
import numpy as np
import numpy.testing as npt

from utilities import crossover, roulette, thresholdLayer, tanhLayer
from GA import createPop, pairPop
from neuralnetwork import NN


# class testcrossover(unittest.TestCase):
# 	def setUp(self):
# 		self.pop  = createPop()
# 		rand_individual = self.pop[random.randint(0, len(self.pop))]
# 		self.ch1 = [rand_individual.wi, rand_individual.wo]
# 		rand_individual = self.pop[random.randint(0, len(self.pop))]
# 		self.ch2 = [rand_individual.wi, rand_individual.wo]
# 		self.o1, self.o2 = crossover(self.ch1, self.ch2, NN)

# 	def testOutputsAreNotShallowCopies(self):
# 		for x in [self.ch1, self.ch2]:
# 			for y in [self.o1, self.o2]:
# 				self.assertNotEqual(id(x), id(y))
# 				for io in range(len(self.ch1)):
# 					self.assertNotEqual(id(x[io]), id(y[io]))

# 	def testInputIsChanged(self):
# 		"""Kinda redundant r may be 0"""
# 		# for x in [self.ch1, self.ch2]:
# 		# 	for y in [self.o1, self.o2]:
# 		# 		# comparison = where(x != y, True, False)
# 		# 		# self.assertTrue(comparison.all())
# 		# 		for io in range(len(self.ch1)):
# 		# 			comparison = where(x[io] != y[io], True, False)
# 		# 			self.assertTrue(comparison.all(), str(comparison))
# 		pass

# 	def testShapeUnchanged(self):
# 		print 'ch1', self.ch1
# 		print 'ch2', self.ch2
# 		print 'o1', self.o1
# 		print 'o2', self.o2

# 		for x in array([self.ch1, self.ch2]):
# 			for y in array([self.o1, self.o2]):
# 				self.assertItemsEqual(x.shape, y.shape)
# 				for io in range(len(self.ch1)):
# 					self.assertItemsEqual(array(x[io]).shape, 
# 						array(y[io]).shape)


# class testroulette(unittest.TestCase):
# 	def setUp(self):
# 		self.pop = createPop()
# 		self.pairedPop = pairPop(self.pop)
# 		self.rankedPop = sorted(self.pairedPop, key=itemgetter(-1), reverse=True) 
# 		self.rankedWeights = [x[0] for x in self.rankedPop]
# 		self.fitnessScores = [x[-1] for x in self.rankedPop]
# 		self.copy = array(self.fitnessScores).copy()
# 		self.index = roulette(self.fitnessScores)

# 	def testFitnessScoresNotChanged(self):
# 		self.assertNotEqual(id(self.copy),
# 			id(self.fitnessScores))
# 		for i in range(len(self.copy)):
# 			self.assertNotEqual(id(self.copy[i]),
# 				id(self.fitnessScores[i]), str(self.copy[i]) + str(i))
# 		self.assertItemsEqual(array(self.copy).shape,
# 			array(self.fitnessScores).shape)


# class testthresholdLayer(unittest.TestCase):
# 	def setUp(self):
# 		self.vals = linspace(-1, 1, 10)
# 		self.out = thresholdLayer(self.vals)
# 		self.target = array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.])

# 	def testOut(self):
# 		comp = self.out - self.target
# 		self.assertFalse(comp.any())

class testtanhLayer(unittest.TestCase):
	def setUp(self):
		self.vals = linspace(-1, 1, 10)
		

	def testAsInputLayer(self):
		out = tanhLayer(self.vals, isInputLayer=True)
		targets = np.tanh(self.vals)
		npt.assert_array_equal(out, targets)
		# self.target = array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.])
		# comp = self.out - self.target
		# self.assertFalse(comp.any())
	
	def testAsNormalLayer(self):
		nextLayerDim = 6
		weights = ones(self.vals.shape, nextLayerDim)


if __name__ == '__main__':
	unittest.main()