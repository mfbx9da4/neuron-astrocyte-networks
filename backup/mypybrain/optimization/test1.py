# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:41:24 2013

Just to try and get things working with a GA


@author: david
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.rl.environments.functions.unimodal import TabletFunction
from pybrain.rl.environments.cartpole.balancetask import BalanceTask, CartPoleEnvironment
from pybrain.optimization import GA
from pybrain.rl.agents import LearningAgent, OptimizationAgent


environment = CartPoleEnvironment()
task = BalanceTask()

nn = buildNetwork(task.outdim, 6, task.indim)

learning_agent = OptimizationAgent(nn, GA())


