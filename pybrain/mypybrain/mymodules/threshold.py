# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:53:26 2013

@author: david
"""


import numpy as np
from pybrain.structure.modules.neuronlayer import NeuronLayer


def threshold(a, t):
    result = []
    for x in a:
        if x >= t:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)


class ThresholdLayer(NeuronLayer):
    """Simple threshold layer produces binary output """
    def __init__(self, dim, name=None, threshold=0.5):
        NeuronLayer.__init__(self, dim, name)
        self.t = threshold

    def _forwardImplementation(self, inbuf, outbuf):
        # could replace with where(a > 0.5, 1, 0)
        #outbuf[:] = threshold(inbuf, self.t)
        outbuf[:] = np.where(inbuf > 0.5, 1., 0.)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr
