import unittest 

import numpy as np
import numpy.testing as npt

from neuralnetwork import NN

class testactivate(unittest.TestCase):
	def setUp(self):
		self.nn = NN()
		self.inputs = [1, 2, 3, 4]
		self.nn.activate(self.inputs)

	def testInputActivation(self):
		targets = self.nn.activate_input(self.inputs)
		npt.assert_array_equal(self.nn.in_activations,
			targets)

	def testHiddenActivation(self):
		# targets = NN.activate_hidden([1, 2, 3, 4])
		# npt.assert_array_equal(self.nn.in_activations,
		# 	targets)
		pass

	def testShapesUnchanged(self):
		pass

class testinitialization(unittest.TestCase):
	def setUp(self):
		self.nn = NN()

	def testShapeOfActivations(self):
		self.assertEqual(self.nn.in_activations.shape,
			(NN.ni, ))
		self.assertEqual(self.nn.hid_activations.shape,
			(NN.nh, ))
		self.assertEqual(self.nn.out_activations.shape,
			(NN.no, ))

	def testTypeOfActivations(self):
		self.assertIsInstance(self.nn.in_activations,
			np.ndarray)
		self.assertIsInstance(self.nn.hid_activations,
			np.ndarray)
		self.assertIsInstance(self.nn.out_activations,
			np.ndarray)

	def testShapeOfWeights(self):
		self.assertEqual(self.nn.weights.shape,
		(NN.ni*NN.nh + NN.nh*NN.no, ))


		


if __name__ == '__main__':
	unittest.main()