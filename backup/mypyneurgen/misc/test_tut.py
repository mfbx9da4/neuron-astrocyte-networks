from tut import *
import unittest

class testbuildSamples(unittest.TestCase):
	"""Tests build samples function"""
	def setUp(self):
		self.all_inputs, self.all_targets = buildSamples()

	def testShapes(self):
		self.assertEqual(array(self.all_inputs).shape, (150 ,4), 
			array(self.all_inputs).shape)
		self.assertEqual(array(self.all_targets).shape, (150, 3), 
			array(self.all_targets).shape)

	def testRanges(self):
		for target in self.all_targets:
			for val in target:
				self.assertIn(val, [1., 0.])
		for inp in self.all_inputs:
			for val in inp:
				self.assertIsInstance(val, float)
				self.assertLessEqual(val, 12.)
				self.assertGreaterEqual(val, 0.1)

class testbuildNetwork(unittest.TestCase):
	"""Tests buildNetwork funciton"""
	def setUp(self):
		inps, trgs = buildSamples()
		self.net = buildIrisNetwork(inps, trgs)

	def testData(self):
		self.assertEqual(array(list(self.net.get_learn_data())).shape, (75,2))
		self.assertEqual(array(list(self.net.get_test_data())).shape, (73,2))

	def testLayers(self):
		pass

	def testThresholdNode(self):
		outLayer = self.net.output_layer	
		outLayer.set_activation_type('threshold')
		self.assertEqual(outLayer.get_node(0).get_activation_type(),
			'threshold')
		vals = [-0.5, 0., 0.6]
		assert len(vals) == outLayer.total_nodes()
		for i in range(len(vals)):
			n = outLayer.get_node(i)
			n.set_value(vals[i])
			n.activate()
		out = outLayer.activations()
		self.assertItemsEqual(where(array(vals) > 0.5, 1., 0.), out,
			'threshold not working' + \
			'\n' + str(vals) + \
			'\n' + str(out) + \
			'\n' + str(where(vals > 0.5, 1., 0.)))

		
if __name__== '__main__':
  unittest.main()


