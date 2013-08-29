import unittest 
from operator import itemgetter
# could use import timeit; timeit.timeit(s) to check slow parts
from pylab import bincount
import numpy as np
import numpy.testing as npt

from GA import createPop, pairPop, evolveNewPop
from utilities import sigmoid
from neuralnetwork import NN


class testinitializer(unittest.TestCase):
	def setUp(self):
		self.net = NN()

	def testActivationShapes(self):
		self.assertEqual(self.net.ai.shape[0], NN.ni)


class testactivate(unittest.TestCase):
	def setUp(self):
		weights_in = -2 + np.random.rand(NN.ni, NN.nh) * 4
		weights_out = -2 + np.random.rand(NN.nh, NN.no) * 4
		self.net1 = NN()
		self.net1.wi = weights_in.copy()
		self.net1.wo = weights_out.copy()
		self.net2 = NN()
		self.net2.wi = weights_in.copy()
		self.net2.wo = weights_out.copy()
		inputs = -2 + np.random.rand(4) * 8
		for i in range(NN.ni):
			self.net1.ai[i] = np.tanh(inputs[i])
		for j in range(NN.nh):
			self.net1.ah[j] = sigmoid(sum([self.net1.ai[i] * self.net1.wi[i][j] for i in range(NN.ni)]))
		for k in range(NN.no):
			self.net1.ao[k] = sum([self.net1.ah[j] * self.net1.wo[j][k] for j in range(NN.nh)])
		self.net1.ao = np.where(self.net1.ao > 0.5, 1.0, 0.0)
		self.net2.activate(inputs)

	def testactivationfunctions(self):
		"""sometimes this will fail when normal output would be 
		an array containig multiple ones"""
		npt.assert_array_equal(self.net1.ai,
			self.net2.ai)
		npt.assert_array_equal(self.net1.ah,
			self.net2.ah)
		npt.assert_array_equal(self.net1.ao,
			self.net2.ao)

	def testOutput(self):
		self.assertEqual(bincount(self.net2.ao.astype('int'))[0], 2)
		self.assertEqual(bincount(self.net2.ao.astype('int'))[1], 1)


class testassignWeights(unittest.TestCase):
	def setUp(self):
		self.pop = createPop()
		paired = pairPop(self.pop, verbose=False)
		ranked = sorted(paired, key=itemgetter(-1), reverse=True)
		self.newpopW = evolveNewPop(ranked)
		for i in range(len(self.pop)):
			self.pop[i].assignWeights(self.newpopW[i])

	def testWeightsAreNotIdentical(self):
		"""Checking to see if assigned weights are a view of source weights
		"""
		for i in range(len(self.newpopW)):
			for j in range(len(self.pop)):
				self.assertIsNot(self.newpopW[i], self.pop[j])
				io = 0
				self.assertIsNot(self.newpopW[i][io], self.pop[j].wi)
				for w_i, w in enumerate(self.newpopW[i][io]):
					self.assertIsNot(w, self.pop[j].wi[w_i])
				io = 1
				self.assertIsNot(self.newpopW[i][io], self.pop[j].wo)
				for w_i, w in enumerate(self.newpopW[i][io]):
					self.assertIsNot(w, self.pop[j].wo[w_i])

	def testWeightsAreEqual(self):
		for i in range(len(self.newpopW)):
			io = 0
			npt.assert_array_equal(
				self.newpopW[i][io],
				self.pop[i].wi)
			io = 1
			npt.assert_array_equal(
				self.newpopW[i][io],
				self.pop[i].wo)

class testsumErrors(unittest.TestCase):
	def setUp(self):
		self.pop = createPop()
		self.inv_err, self.perc_err = self.pop[0].sumErrors()

	def testOutput(self):
		self.assertIsInstance(self.perc_err, float)
		self.assertLess(self.perc_err, 100.00001)
		self.assertGreater(self.perc_err, -0.0000001)

















if __name__ == '__main__':
	unittest.main()