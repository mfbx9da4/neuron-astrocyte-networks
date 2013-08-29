# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 00:26:47 2013

@author: david
"""

import sys
sys.path.append('/home/david/Dropbox/programming/python/ann/mypybrain')
import unittest

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pylab import *

from mymodules.threshold import ThresholdLayer



class testResult(unittest.TestCase):
    def setUp(self):
        self.nn = buildNetwork(4, 6, 3, inclass=TanhLayer,
                          hiddenclass=TanhLayer,
                          outclass=ThresholdLayer)
        self.nn.activate((rand(4)*20)-10)
        for grand_iter in xrange(iterations):
            errors.append(rand(1))
            weights.append(rand(24*3))
            activations.append([rand(6), rand(4), rand(3))
        #[errors, weights, activations]
    
          