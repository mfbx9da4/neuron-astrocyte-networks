import random

from pylab import array, ones, zeros, rand
import numpy as np

from utilities import thresholdLayer, tanhLayer
from iris import pat as train_pat


class NN(object):
    graphical_error_scale = 100
    pop_size = 100
    mutation_rate = 0.1
    crossover_rate = 0.8
    ni = 4
    nh = 6
    no = 1
    pat = train_pat
    eliteN = int(pop_size * 0.15)


    def __init__(self):
        self.in_activations = ones(NN.ni)
        self.hid_activations = ones(NN.nh)
        self.out_activations = ones(NN.no)
        self.weights = rand(NN.ni*NN.nh + NN.nh*NN.no)

    def activate_input(self, inputs):
        assert len(inputs) == NN.ni, 'incorrect number of inputs'
        self.in_activations[:] = tanhLayer(inputs, isInputLayer=True)

        # activate_hidden = tanhLayer
        # activate_output = thresholdLayer

    def activate(self, inputs):
        self.activate_input(inputs)
        # for j in range(NN.nh):
        #     self.hid_activations[j] = sigmoid(sum([ self.in_activations[i]*self.wi[i][j] for i in range(NN.ni) ]))
        # for k in range(NN.no):
        #     self.out_activations[k] = sigmoid(sum([ self.hid_activations[j]*self.wo[j][k] for j in range(NN.nh) ]))
        return self.out_activations

    def printWeights(self):
        print 'Input weights:'
        for i in range(NN.ni):
            print self.wi[i]
        print
        print 'Output weights:'
        for j in range(NN.nh):
            print self.wo[j]
        print ''

    def test(self, patterns):
        results, targets = [], []
        for p in patterns:
            inputs = p[0]
            rounded = [round(i) for i in self.activate(inputs)]
            if rounded == p[1]: 
                result = '+++++'
            else: 
                result = '-----'
            print '%s %s %s %s %s %s %s' % ('Inputs:', p[0], '-->', str(self.activate(inputs)).rjust(65), 'Target', p[1], result)
            output_activations = self.activate(inputs)
            if len(output_activations) == 1:
                results.append(output_activations[0])
            targets += p[1]
        return results, targets

    def sumErrors(self):
        mse = 0.0
        for p in NN.pat:
            inputs = p[0]
            targets = p[1]
            self.activate(inputs)
            mse += self.calcMse(targets)
        inverr = 1.0 / mse
        return inverr

    def calcMse(self, targets):
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.out_activations[k]) ** 2
        return error

    def assignWeights(self, new_weights):
        io = 0
        for i in range(NN.ni):
            for j in range(NN.nh):
                self.wi[i][j] = new_weights[io][i][j]
        io = 1
        for j in range(NN.nh):
            for k in range(NN.no):
                self.wo[j][k] = new_weights[io][j][k]

    def testWeights(self, weights, I):
        same = []
        io = 0
        for i in range(NN.ni):
            for j in range(NN.nh):
                if self.wi[i][j] != weights[I][io][i][j]:
                    same.append(('I',i,j, round(self.wi[i][j],2),round(weights[I][io][i][j],2),round(self.wi[i][j] - weights[I][io][i][j],2)))

        io = 1
        for j in range(NN.nh):
            for k in range(NN.no):
                if self.wo[j][k] !=  weights[I][io][j][k]:
                    same.append((('O',j,k), round(self.wo[j][k],2),round(weights[I][io][j][k],2),round(self.wo[j][k] - weights[I][io][j][k],2)))
        if same:
            print same

