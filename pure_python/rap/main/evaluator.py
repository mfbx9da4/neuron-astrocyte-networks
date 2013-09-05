from numpy import array
from numpy.random import permutation
from pybrain.utilities import percentError

from utilities import OutToClass


class Evaluator(object):
    def __init__(self, net, ds, verbose=False):
        self.tester_net = net
        self.trn_d = ds[0]
        self.tst_d = ds[1]
        self.verbose = verbose

    def testOnTrainData(self, new_weights):
        self.assignWeights(new_weights) 
        outs, trgs = self.activateOnAllSamples(return_targets=True)
        outs, trgs = OutToClass(outs, trgs)
        pe = percentError(outs, self.trn_d['class'])
        if self.verbose:
            print pe
        return pe
    
    def assignWeights(self, new_weights):
        self.tester_net.resetAstros()
        self.weights = new_weights

    def activateOnAllSamples(self, return_targets=False):
        shuffled = self.shuffleData()
        outs = [self.tester_net.activate(i) for i, t in shuffled]
        self.tester_net.resetAstros()
        if self.verbose:  printOutputs(shuffled, outs)
        if return_targets:
            targets = [x[1] for x in shuffled]
            return outs, targets
        return outs

    def shuffleData(self):
        ins = self.trn_d['input'].copy()
        tgs = self.trn_d['target'].copy()
        order = permutation(len(ins))
        shuffled = []
        for i in order:
            shuffled.append((ins[i], tgs[i]))
        return shuffled

def printOutputs(shuffled, outs):
    for p, o in zip(shuffled, outs):
        i = p[0]
        t = p[1]
        print i, o, t, '++++' if list(o) == list(t) else '--'
    raw_input('Press enter to continue to next iter')