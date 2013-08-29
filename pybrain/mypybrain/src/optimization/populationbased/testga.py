import unittest

class testtournamentSelect(unittest.TestCase):
	def setUp(self):
	    nn = createIrisNN()
	    ga = GA(percError, nn, minimize=True, desiredEvaluation=0.0, 
		    verbose=True, mutationProb=0.1, populationSize=150,
		    storeAllEvaluations=True, elitism=True, tournament=True)


if __name__ == '__main__':
	unittest.main()