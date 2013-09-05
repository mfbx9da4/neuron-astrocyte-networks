# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 10:29:04 2013

test for self-made GA PP astrocyteLayer

@author: david
"""
import unittest

from pylab import linspace, append, array, zeros, ones, rand, empty
import numpy.testing as npt 

from neuralnetwork import NN
from astrocyte_layer import *


class testFunctions(unittest.TestCase):
    def testRoundActivationsUpOrDown(self): 
        result = roundActivationsUpOrDown(linspace(-1,1,10))
        target = append(zeros(5)-1, ones(5))
        self.assertEqual((result-target).any(), False)


class testUpdate(unittest.TestCase):
    def setUp(self):
        self.nn = NN()
        self.in_to_hidden = self.nn.wi
        self.hidAstroL = AstrocyteLayer(self.nn.ah, self.in_to_hidden)

    def testUpdateNeuronCounters(self):
        self.hidAstroL.neur_activs = linspace(-1, 1, NN.nh)
        self.hidAstroL.updateNeuronCounters()
        npt.assert_array_equal(self.hidAstroL.neur_counters,
            array([-1, -1, -1, 1, 1, 1]))

    def testUpdateAstrocyteActivations(self):
        result = map(self.hidAstroL._checkIfAstroActive, 
            map(int, linspace(-3, 3, 8)))
        trgt = array([-1, -1, -1, 0, 0, 1, 1, 1])
        npt.assert_array_equal(result, trgt)

    def testPerformAstrocyteActions(self):
        self.hidAstroL.neur_in_ws[:] = ones(24).reshape(NN.ni, NN.nh)
        """Assuming we are at the fourth iter of minor iters and that astro 
        parameters are reset after each input pattern, neur counters can only 
        be in [4, -4, 2, -2, 0]"""
        self.hidAstroL.neur_counters[:] = array([4, -4, 2, -2, 0, 0])
        self.hidAstroL.remaining_active_durs[:] = array([3, -3, 2, -2, 0, 0])
        self.hidAstroL.astro_statuses[:] = array([1, -1, 1, -1, 0, 0])
        self.hidAstroL.performAstroActions()
        target_weights = empty([NN.ni, NN.nh])
        target_weights[:,0] = 1.25
        target_weights[:,1] = .5
        target_weights[:,2] = 1.25
        target_weights[:,3] = .5
        target_weights[:,4] = 1.
        target_weights[:,5] = 1.
        npt.assert_array_equal(self.in_to_hidden, target_weights)
        target_activations = array([1, -1, 1, -1, 0, 0])
        npt.assert_array_equal(target_activations, self.hidAstroL.astro_statuses)
        target_remaining_active_durs = array([2, -2, 1, -1, 0, 0])
        npt.assert_array_equal(target_remaining_active_durs, 
            self.hidAstroL.remaining_active_durs)

    def testObjectActivationsAreUpdated(self):
        self.nn.activate(rand(4)*4)
        a = self.hidAstroL.neur_in_ws
        npt.assert_array_equal(self.in_to_hidden, a) 

    def testObjectWeightsAreUpdated(self):
        # don't understand this test don't think i need it
        # it is a bit random lol
        target = empty([NN.ni, NN.nh])  
        for j in range(len(self.nn.ah)):
            self.hidAstroL.neur_in_ws[:,j] = 10 ** j
            target[:,j] = 10 ** j
        npt.assert_array_equal(self.in_to_hidden, self.hidAstroL.neur_in_ws)
        npt.assert_array_equal(self.in_to_hidden, target)
                                                                                


if __name__== '__main__':
    unittest.main()