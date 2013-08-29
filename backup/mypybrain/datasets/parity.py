# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:58:41 2013

Parity dataset

@author: david
"""
from pybrain.datasets import ClassificationDataSet
import random

def parity(k, n, name="parity"):
    """Return a DataSet with n k-bit examples of the parity problem:
    k random bits followed by a 1 if an odd number of bits are 1, else 0."""
    examples = []
    for i in range(n):
        bits = [random.choice([0, 1]) for i in range(k)]
        bits.append(sum(bits) % 2)
        examples.append(bits)
    return examples

class ParityDataSet(ClassificationDataSet):
    """Odd number of bits of value 1, target is 1, else target is 0"""
    def __init__(self, nbits=4, nsamples=50):
        ClassificationDataSet.__init__(self, nbits, target=1, nb_classes=1)
        samples = parity(nbits, nsamples)
        for x in samples:
            self.addSample(x[:-1], [x[-1]])
        

