import unittest 
from operator import itemgetter

import numpy.testing as npt

from GA import createPop, pairPop, evolveNewPop

class testassignWeights(unittest.TestCase):
	def setUp(self):
		self.pop = createPop()
		paired = pairPop(self.pop, verbose=False)
		ranked = sorted(paired, key=itemgetter(-1), reverse=True)
		self.newpopW = evolveNewPop(ranked)
		for i in range(len(self.pop)):
			self.pop[i].assignWeights(self.newpopW[i])


	def testWeightsAreNotIdentical(self):
		for i in range(len(self.newpopW)):
			for j in range(len(self.pop)):
				self.assertNotEqual(id(self.newpopW[i]),
					id(self.pop[j]),
					'individual %d\'s weights are a view of ranked' % i +
					'weights %d' % j)	
				io = 0
				self.assertNotEqual(id(self.newpopW[i][io]),
					id(self.pop[j].wi),
					'individual %d\'s %d weights are a view ' 
					% (i, io) + 'of ranked weights %d' % j)
				io = 1
				self.assertNotEqual(id(self.newpopW[i][io]),
					id(self.pop[j].wo),
					'individual %d\'s %d weights are a view ' 
					% (i, io) + 'of ranked weights %d' % j)

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




if __name__ == '__main__':
	unittest.main()