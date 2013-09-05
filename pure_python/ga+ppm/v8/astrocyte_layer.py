# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:05:30 2013

Porto-Pazos model of astrocyte for self-made GA

self.update() is called each astrocyte processing iteration (each n
of m iters)

Astrocyte updates the input weights to its associated astrocyte



@author: david
"""

from scipy import zeros, sign, where
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
    def __init__(self, layer_activations, layer_weights, **kwargs):
        """Layer is the neur layer, conncection is the input 
        connection to this layer and has the property params which holds
        the values of the weights
        """
        params = {
            'astro_thresh' : 3,
            'astro_dur' : 4, 
            'astro_processing_iters' : 6,
            'incr' : 0.25, 
            'decr' : 0.5}
        for k in kwargs.keys():
            params[k] = kwargs[k]
        """astro settings"""
        self.dim = len(layer_activations)
        self.astro_thresh = params['astro_thresh']
        self.astro_dur = params['astro_dur']
        self.astro_processing_iters = params['astro_processing_iters']
        self.incr_percent = params['incr']
        self.decr_percent = params['decr']
        """Associated neur layer and weights"""
        self.neur_activs = layer_activations
        """Initialize astro parameters"""
        self.neur_counters = zeros(self.dim, dtype=int)
        self.remaining_active_durs = zeros(self.dim, dtype=int)
        self.astro_statuses = zeros(self.dim, dtype=int)

    def update(self):
        """
        1. Astrocytic neur counter (ANC) counts the decision of associated neur.
        2. If ANC reaches positive or negative thresh, astro is updated to be
        positively (+1) or negatively (-1) activated for a fixed dur of
        iterations, respectively.
        3. If astro is activated, weights are updated and remaing active
        durs are decred.
        * could get update to return counters, activations, remaining_active_durs
        and statuses
        * could make an InputAstroL, HidAstroLayer and OutAstroLayer. The only 
        that would change is I would initialize differently and update would be
        different ie hid has two layers and modifies them both at the same time  
        """
        self.updateNeuronCounters()
        self.updateAstroActivations()
        self.performAstroActions()

    def updateNeuronCounters(self):
        self.neur_counters += where(self.neur_activs > 0, 1, -1)

    def updateAstroActivations(self):
        self.remaining_active_durs = map(self._checkIfthresh,
            self.neur_counters)
        self.astro_statuses = map(self._checkIfAstroActive,
            self.remaining_active_durs)

    def _checkIfthresh(self, counter):
        # not very pythonic
        assert counter <= self.astro_processing_iters and counter >= -self.astro_processing_iters
        if counter >= self.astro_thresh:
            return self.astro_dur
        elif counter <= -self.astro_thresh:
            return -self.astro_dur
        else:
            return 0

    def _checkIfAstroActive(self, remaining_dur):
        assert remaining_dur <= self.astro_dur \
        and remaining_dur >= -self.astro_dur
        if remaining_dur > 0:
            return 1
        elif remaining_dur < 0:
            return -1
        else:
            return 0

    def reset(self):
        self.neur_counters = zeros(self.dim, dtype=int)
        self.remaining_active_durs = zeros(self.dim, dtype=int)
        self.astro_statuses = zeros(self.dim, dtype=int)


class InAstroLayer(AstrocyteLayer):
    def __init__(self, layer_activations, input_ws, output_ws, **kwargs):
        AstrocyteLayer.__init__(self, layer_activations)
        self.in_ws = input_ws
        self.out_ws = output_ws

    def performAstroActions(self):
        for j, active in enumerate(self.astro_statuses):
            assert active in [-1, 0, 1]
            if active == 1:
                assert sign(self.remaining_active_durs[j]) == 1
                self.out_ws[j] += self.out_ws[j] * self.incr_percent
                self.remaining_active_durs[j] -= 1
            elif active == -1:
                assert sign(self.remaining_active_durs[j]) == -1
                self.out_ws[j] += self.out_ws[j] * -self.decr_percent
                self.remaining_active_durs[j] += 1


class HidAstroLayer(AstrocyteLayer):
    def __init__(self, layer_activations, input_ws, output_ws, **kwargs):
        AstrocyteLayer.__init__(self, layer_activations)
        self.in_ws = input_ws
        self.out_ws = output_ws

    def performAstroActions(self):
        for j, active in enumerate(self.astro_statuses):
            assert active in [-1, 0, 1]
            if active == 1:
                assert sign(self.remaining_active_durs[j]) == 1
                self.in_ws[:,j] += self.in_ws[:,j] * self.incr_percent
                self.out_ws[j] += self.out_ws[j] * self.incr_percent
                self.remaining_active_durs[j] -= 1
            elif active == -1:
                assert sign(self.remaining_active_durs[j]) == -1
                self.in_ws[:,j] += self.in_ws[:,j] * -self.decr_percent
                self.out_ws[j] += self.out_ws[j] * -self.decr_percent
                self.remaining_active_durs[j] += 1


class OutAstroLayer(AstrocyteLayer):
    def __init__(self, layer_activations, input_ws, output_ws, **kwargs):
        AstrocyteLayer.__init__(self, layer_activations)
        self.in_ws = input_ws
        self.out_ws = output_ws

    def performAstroActions(self):
        for j, active in enumerate(self.astro_statuses):
            assert active in [-1, 0, 1]
            if active == 1:
                assert sign(self.remaining_active_durs[j]) == 1
                self.in_ws[:,j] += self.in_ws[:,j] * self.incr_percent
                self.remaining_active_durs[j] -= 1
            elif active == -1:
                assert sign(self.remaining_active_durs[j]) == -1
                self.in_ws[:,j] += self.in_ws[:,j] * -self.decr_percent
                self.remaining_active_durs[j] += 1


