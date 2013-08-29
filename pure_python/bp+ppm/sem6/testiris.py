import unittest
import random

from pylab import array

import iris


class testrandomIrisData(unittest.TestCase):
	def setUp(self):
		self.split = random.random()
		self.train, self.test = iris.randomIrisData(self.split)

	def testShape(self):
		self.assertEqual(len(self.train) + len(self.test),
			150)
		self.assertAlmostEqual(len(self.train) / 100.,
			(150 - int(150 * self.split)) / 100.,
			2)

	def testSplitOfCats(self):
		train_cats = list(array(self.train)[:,2])
		self.assertTrue(train_cats.count(0) == train_cats.count(1) == train_cats.count(2))
		







if __name__ == '__main__':
	unittest.main()