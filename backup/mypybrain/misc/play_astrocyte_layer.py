# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:05:30 2013

Porto-Pazos model of astrocyte

self.update() is called each astrocyte processing iteration (each n
of m iters)

Astrocyte updates the input weights to its associated astrocyte



@author: david
"""

from scipy import zeros, sign
from itertools import *


def roundActivationsUpOrDown(array):
    return map(roundActivationUpOrDown, array)


def roundActivationUpOrDown(x):
    assert x <= 1 and x >= -1
    if x >= 0:
        return 1
    else:
        return -1


class AstrocyteLayer(object):
    def __init__(self, layer, connection, astrocyte_threshold=3,
                 astrocyte_duration=4, astrocyte_processing_iters=6,
                 incr=0.25, decr=0.5):
        """Layer is the neuronal layer, conncection is the input 
        connection to this layer and has the property params which holds
        the values of the weights
        """
        """Astrocyte settings"""
        self.dim = layer.dim
        self.astrocyte_threshold = astrocyte_threshold
        self.astrocyte_duration = astrocyte_duration
        self.astrocyte_processing_iters = astrocyte_processing_iters
        self.increment_percent = incr
        self.decrement_percent = decr
        """Associated neuron layer and weights"""
        self.neuronal_layer = layer
        self.neuronal_input_connection = connection
        """Initialize astrocyte parameters"""
        self.neuron_counters = zeros(self.dim, dtype=int)
        self.remaining_active_durations = zeros(self.dim, dtype=int)
        self.astrocyte_statuses = zeros(self.dim, dtype=int)

    def update(self):
        """
        1. Astrocytic neuron counter (ANC) counts the decision of associated neuron.
        2. If ANC reaches positive or negative threshold, astrocyte is updated to be
        positively (+1) or negatively (-1) activated for a fixed duration of
        iterations, respectively.
        3. If astrocyte is activated, weights are updated and remaing active
        durations are decremented.
        """
        self.updateNeuronCounters()
        self.updateAstrocyteActivations()
        self.performAstrocyteActions()

    def updateNeuronCounters(self):
        neuronal_activations, = self.neuronal_layer.outputbuffer
        self.neuron_counters += roundActivationsUpOrDown(neuronal_activations)

    def updateAstrocyteActivations(self):
        self.remaining_active_durations = map(self._checkIfThreshold,
                                              self.neuron_counters)
        self.astrocyte_statuses = map(self._checkIfAstrocyteActive,
                                    self.remaining_active_durations)

    def performAstrocyteActions(self):
        i = len(self.neuronal_input_connection.params)/self.dim
        for j, active in enumerate(self.astrocyte_statuses):
            J = j*i
            assert active in [-1, 0, 1]
            if active == 1:
                # NEED TO CHECK _setParameters and _params method
                assert sign(self.remaining_active_durations[j]) == 1
                self.neuronal_input_connection.params[J:J+i] += \
                  self.neuronal_input_connection.params[J:J+i]*self.increment_percent
                self.remaining_active_durations[j] -= 1
            elif active == -1:
                assert sign(self.remaining_active_durations[j]) == -1
                self.neuronal_input_connection.params[J:J+i] += \
                  self.neuronal_input_connection.params[J:J+i]*-self.decrement_percent
                self.remaining_active_durations[j] += 1

    def _checkIfThreshold(self, counter):
        assert counter <= self.astrocyte_processing_iters\
          and counter >= -self.astrocyte_processing_iters
        if counter >= self.astrocyte_threshold:
            return self.astrocyte_duration
        elif counter <= -self.astrocyte_threshold:
            return -self.astrocyte_duration
        else:
            return 0

    def _checkIfAstrocyteActive(self, remaining_dur):
        assert remaining_dur <= self.astrocyte_duration \
        and remaining_dur >= -self.astrocyte_duration
        if remaining_dur > 0:
            return 1
        elif remaining_dur < 0:
            return -1
        else:
            return 0

    def reset(self):
        self.neuron_counters = zeros(self.dim, dtype=int)
        self.remaining_active_durations = zeros(self.dim, dtype=int)
        self.astrocyte_statuses = zeros(self.dim, dtype=int)
