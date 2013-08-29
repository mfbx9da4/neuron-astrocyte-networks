import sys

from pybrain.optimization.populationbased.ga import GA

from net import NeuroAstroNet
from utilities import xorDataSet
from evaluator import Evaluator


def runTrial(iters):
    pass
def
 main():
    ds_label, trials, iters = getArgs()
    net, ds = buildNetworkAndDs(ds_label)
    ev = Evaluator(net, ds, verbose=False)
    ga = GA(ev.testOnTrainData, net.weights, minimize=True,
        populationSize=150, elitism=True)
    for t in range(trials):
        for i in range(iters):
            print i, ga.learn(0)[1]



def buildNetworkAndDs(ds):
    if ds == 'xor':
        net = NeuroAstroNet(2, 3, 1)
        ds = (xorDataSet(), xorDataSet())
    else:
        raise NotImplementedError
    return net, ds

def getArgs():
    len_args = len(sys.argv)
    ds = sys.argv[1] if len_args > 1 else 'xor'
    trials = int(sys.argv[2]) if len_args > 2 else 1
    iters = int(sys.argv[3]) if len_args > 3 else 4
    return ds, trials, iters



if __name__ == '__main__':
    main()