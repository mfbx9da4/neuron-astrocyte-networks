import unittest
from GA import createPop, pairPop, NN, rankPop, itemgetter
from GA import evolveNewPop, selectTwoIndividuals

from pylab import where, array

class testcreatePop(unittest.TestCase):
	
	def setUp(self):
		self.pop = createPop()

	def testType(self):
		for ind in self.pop:
			self.assertIsInstance(ind, NN)

	def testWeightsAreNotTheSame(self):
		"""Compares each weight against all others but could be sped
		up don't need to compare twice"""
		for i in range(len(self.pop)):
			for j in range(len(self.pop)):
				if i != j:
					wi1 = array(self.pop[i].wi)
					wi2 = array(self.pop[j].wi)
					comparisons = where( wi1 == wi2, True, False)
					for c in comparisons:
						self.assertFalse(c.all())					

					wo1 = array(self.pop[i].wo)
					wo2 = array(self.pop[j].wo)
					comparisons = where( wo1 == wo2, True, False)
					for c in comparisons:
						self.assertFalse(c.all())

	def testShapeOfInputWeights(self):
		for ind in self.pop:
			self.assertEqual(array(ind.wi).shape,
				(NN.ni, NN.nh))

	def testShapeOfOutputWeights(self):
		for ind in self.pop:
			self.assertEqual(array(ind.wo).shape,
				(NN.nh, NN.no))


class testpairPop(unittest.TestCase):
	"""
	paired pop is zip(weights, errors, fitnesses)
	accessed in the order:
		pairedPop[individual][weights][input/output weights]
	"""
	def setUp(self):
		self.pop = createPop()
		self.pairedPop = pairPop(self.pop)

	def testWeightsAreACopy(self):
		for i in range(len(self.pop)):
			self.assertNotEqual(id(self.pop[i].wi),
				id(self.pairedPop[i][0][0]), 
				'input weights, ind ' + str(i) )
			self.assertNotEqual(id(self.pop[i].wo),
				id(self.pairedPop[i][0][1]),
				'output weights, ind ' + str(i))

	def testShapeOfInputWeights(self):
		for ind in self.pairedPop:
			self.assertEqual(array(ind[0][0]).shape,
				(NN.ni, NN.nh))

	def testShapeOfOutputWeights(self):
		for ind in self.pairedPop:
			self.assertEqual(array(ind[0][1]).shape,
				(NN.nh, NN.no))

class testrankPop(unittest.TestCase):
	def setUp(self):
		# need to test that rankedPop is ordered in descending order
		pass

class testevolveNewPop(unittest.TestCase):
	"""
	rankedPop is zip(weights, errors, fitnesses) ordered in descending 
	order of fitness
	"""
	def setUp(self):
		self.pop = createPop()
		self.pairedPop = pairPop(self.pop)
		self.rankedPop = sorted(self.pairedPop, key=itemgetter(-1), reverse=True) 
		self.rankedWeights = [x[0] for x in self.rankedPop]
		self.fitnessScores = [x[-1] for x in self.rankedPop]
		self.newpopW = evolveNewPop(self.rankedPop)
	
	def testShapeOfInputWeights(self):
		for ind in self.pairedPop:
			self.assertEqual(array(ind[0][0]).shape,
				(NN.ni, NN.nh))

	def testShapeOfOutputWeights(self):
		for ind in self.pairedPop:
			self.assertEqual(array(ind[0][1]).shape,
				(NN.nh, NN.no))

	def testNotCopiesOfRankedPop(self):
		for i in range(len(self.newpopW)):
			for j in range(len(self.rankedWeights)):
				self.assertNotEqual(id(self.newpopW[i]),
					id(self.rankedWeights[j]),
					'individual %d\'s weights are a view of ranked' % i +
					'weights %d' % j)	
				for io in range(len(self.newpopW[i])):
					self.assertNotEqual(id(self.newpopW[i][io]),
						id(self.rankedWeights[j][io]),
						'individual %d\'s %d weights are a view ' 
						% (i, io) + 'of ranked weights %d' % j)

	def testElitism(self):
		for i in range(NN.eliteN):
			for io in range(2):
				shouldBeZeros = self.rankedWeights[i][io] - self.newpopW[i][io]
				self.assertFalse(shouldBeZeros.any())
	
	def testLengthOfNewPop(self):
		self.assertEqual(len(self.newpopW), NN.pop_size)

	def testShapeOfNewPop(self):
		oldshape = array(self.rankedWeights).shape	
		newshape = array(self.newpopW).shape	
		self.assertEqual(oldshape, newshape)
		for i in range(len(self.pop)):
			for io in range(len(self.rankedWeights[i])):
				assert io <= 1
				self.assertEqual(
					array(self.rankedWeights[i][io]).shape,
					array(self.newpopW[i][io]).shape)

class testselectTwoIndividuals(unittest.TestCase):
	def setUp(self):
		self.pop = createPop()	
		self.pairedPop = pairPop(self.pop)
		self.rankedPop = sorted(self.pairedPop, key=itemgetter(-1), reverse=True) 
		self.rankedWeights = [x[0] for x in self.rankedPop]
		self.fitnessScores = [x[-1] for x in self.rankedPop]
		self.ch1, self.ch2 = selectTwoIndividuals(self.fitnessScores, self.rankedWeights)

	def testChromosomesAreNotShallowCopies(self):
		for i in range(len(self.rankedWeights)):
			self.assertNotEqual(
				id(self.ch1),
				id(self.rankedWeights[i]))
			self.assertNotEqual(
				id(self.ch2),
				id(self.rankedWeights[i]))
			for io in range(len(self.rankedWeights[i])):
				assert io <= 1
				self.assertNotEqual(
					id(self.ch1[io]),
					id(self.rankedWeights[i][io]))
				self.assertNotEqual(
					id(self.ch2[io]),
					id(self.rankedWeights[i][io]))




if __name__ == '__main__':
	unittest.main()