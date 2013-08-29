# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 10:29:04 2013

@author: david
"""
import unittest

from pylab import linspace, append, array, zeros, ones, rand
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules.tanhlayer import TanhLayer

from astrocyte_layer import *

class testFunctions(unittest.TestCase):
  def testRoundActivationsUpOrDown(self):
    result = roundActivationsUpOrDown(linspace(-1,1,10))
    target = append(zeros(5)-1, ones(5))
    self.assertEqual((result-target).any(), False)

class testUpdate(unittest.TestCase):
  def setUp(self):
    self.nn = buildNetwork(4,6,3, bias=False, hiddenclass=TanhLayer, 
                     outclass=TanhLayer)
    self.nn.sortModules()
    self.in_to_hidden, = self.nn.connections[self.nn['in']]
    self.hiddenAstroLayer = AstrocyteLayer(self.nn['hidden0'], 
                                           self.in_to_hidden)
    
  def testUpdateNeuronCounters(self):
    self.hiddenAstroLayer.neuronal_input_connection.params[:] = ones(24)
    self.assertItemsEqual(self.hiddenAstroLayer.neuron_counters, zeros(6))
    
  def testUpdateAstrocyteActivations(self):
    result = map(self.hiddenAstroLayer._checkIfAstrocyteActive, 
                 map(int, linspace(-3, 3, 8)))
    trgt = array([-1, -1, -1, 0, 0, 1, 1, 1])
    self.assertEqual((result-trgt).any(), False, 
                     str('not equal \n%s\nx%s' % (str(result), str(trgt))))

  def testPerformAstrocyteActions(self):
    self.hiddenAstroLayer.neuronal_input_connection.params[:] = ones(24)
    """Assuming we are at the fourth iter of minor iters and that astrocyte 
    parameters are reset after each input pattern, neuron counters can only 
    be in [4, -4, 2, -2, 0]"""
    self.hiddenAstroLayer.neuron_counters[:] =            array([4, -4, 2, -2, 0, 0])
    self.hiddenAstroLayer.remaining_active_durations[:] = array([3, -3, 2, -2, 0, 0])
    self.hiddenAstroLayer.astrocyte_statuses[:] =         array([1, -1, 1, -1, 0, 0])
    self.hiddenAstroLayer.performAstrocyteActions()
    target_weights = array([[1.25]*4, [0.5]*4, [1.25]*4, [0.5]*4, [1.]*4, 
                            [1.]*4]).reshape(24)
    self.assertItemsEqual(self.in_to_hidden.params, target_weights,
                    str('not equal \n%s\nx%s' % (str(self.in_to_hidden.params), 
                                                    str(target_weights))))
    target_activations = array([1, -1, 1, -1, 0, 0])
    self.assertItemsEqual(target_activations, 
                          self.hiddenAstroLayer.astrocyte_statuses)
                          
    
    target_remaining_active_durations = array([2, -2, 1, -1, 0, 0])
    self.assertItemsEqual(target_remaining_active_durations, 
                          self.hiddenAstroLayer.remaining_active_durations)
        
  def testObjectActivationsAreUpdated(self):
    self.nn.activate(rand(4)*4)
    a = self.hiddenAstroLayer.neuronal_input_connection.params
    self.assertEqual((self.in_to_hidden.params-a).any(), False, 
                     str('not equal \n%s\nx%s' % (str(self.in_to_hidden.params), 
                                                    str(a))))
  def testObjectWeightsAreUpdated(self):
    i = self.nn['in'].dim
    for j in range(self.nn['hidden0'].dim):
      J = j*i
      self.hiddenAstroLayer.neuronal_input_connection.params[J:J+i] =  10**j
    a = self.hiddenAstroLayer.neuronal_input_connection.params
#    print in_to_hidden.params.reshape(6,4), len(in_to_hidden.params)
    target = array([ [10**j]*self.nn['in'].dim \
      for j in range(self.nn['hidden0'].dim)]).reshape(24)
    self.assertEqual((self.in_to_hidden.params-a).any(), False, 
                       str('not equal \n%s\nx%s' % (str(self.in_to_hidden.params), 
                                                    str(a))))
    self.assertEqual((self.in_to_hidden.params-target).any(), False, 
                       str('not equal \n%s\nx%s' % (str(self.in_to_hidden.params), 
                                                    str(target))))                                                    
                                                    
  def tearDown(self):
    del self.hiddenAstroLayer
    del self.nn
    del self.in_to_hidden





if __name__== '__main__':
  unittest.main()