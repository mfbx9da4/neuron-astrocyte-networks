#!/usr/bin/env python
"""
Illustrating the interface of black-box optimizers on a few simple problems:
- how to initialize when:
    * optimizing the parameters for a function
    * optimizing a neural network controller for a task
- how to set meta-parameters
- how to learn
- how to interrupt learning and continue where you left off
- how to access the information gathered during learning
"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array

from pybrain.optimization import * #@UnusedWildImport
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.rl.environments.functions.unimodal import TabletFunction
from pybrain.rl.environments.shipsteer.northwardtask import GoNorthwardTask
from pybrain.rl.environments.cartpole.balancetask import BalanceTask, CartPoleEnvironment
from pybrain.tools.shortcuts import buildNetwork


algo = GA
f = TabletFunction(2)
x0 = [2.1, 4]
l = algo(f, x0)
l.learn(1)

# a very similar interface can be used to optimize the parameters of a Module
# (here a neural network controller) on an EpisodicTask
task = BalanceTask()
nnet = buildNetwork(task.outdim, 2, task.indim)
l = algo(task, nnet)


# Finally you can set storage settings and then access all evaluations made
# during learning, e.g. for plotting:
l = algo(f, x0, storeAllEvaluations = True, storeAllEvaluated = True, maxEvaluations = 150)
for i in range(150):
    print l.learn()
try:
    import pylab
    pylab.plot(map(abs,l._allEvaluations))
    pylab.semilogy()
    pylab.show()
except ImportError, e:
    print 'No plotting:', e

