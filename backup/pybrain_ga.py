import random
import sys
import os
import datetime
from operator import itemgetter

from pybrain.datasets.classification import ClassificationDataSet
from pybrain.optimization.populationbased.ga import GA
from pybrain.supervised.trainers.backprop import BackpropTrainer as bp
# from mypybrain.src.optimization.populationbased.ga import GA
from mypybrain.datasets.parity import ParityDataSet
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.utilities import percentError
from scipy import std, mean, array, bincount, where, argmax
import numpy as np

from myangn.sem6.iris import pybrainData
from mypybrain.mymodules.threshold import ThresholdLayer
from GA_NN.v7.utilities import OutToClass

class Experiment(object):
    def __init__(self, **kw):
        self.params = kw
        self.dataset = None
        self.trn_data = None
        self.tst_data = None 
        self.nn = None
        self.net_settings = None
        self.results = {'trn_per_err':[], 'tst_per_err':[]}

    def buildXor(self):
        self.params['dataset'] = 'XOR'
        d = ClassificationDataSet(2)
        d.addSample([0., 0.], [0.])
        d.addSample([0., 1.], [1.])
        d.addSample([1., 0.], [1.])
        d.addSample([1., 1.], [0.])
        d.setField('class', [[0.], [1.], [1.], [0.]])
        self.trn_data = d
        self.tst_data = d
        global trn_data
        trn_data = self.trn_data
        nn = FeedForwardNetwork()
        inLayer = TanhLayer(2, name='in')
        hiddenLayer = TanhLayer(3, name='hidden0')
        outLayer = ThresholdLayer(1, name='out')
        nn.addInputModule(inLayer)
        nn.addModule(hiddenLayer)
        nn.addOutputModule(outLayer)
        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outLayer)
        nn.addConnection(in_to_hidden)
        nn.addConnection(hidden_to_out)
        nn.sortModules()
        nn.randomize()
        self.net_settings = str(nn.connections)
        self.nn = nn

    def buildParity(self):
        self.params['dataset'] = 'parity'
        self.trn_data = ParityDataSet(nsamples=75)
        self.trn_data.setField('class', self.trn_data['target'])
        self.tst_data = ParityDataSet(nsamples=75)
        global trn_data
        trn_data = self.trn_data
        nn = FeedForwardNetwork()
        inLayer = TanhLayer(4, name='in')
        hiddenLayer = TanhLayer(6, name='hidden0')
        outLayer = ThresholdLayer(1, name='out')
        nn.addInputModule(inLayer)
        nn.addModule(hiddenLayer)
        nn.addOutputModule(outLayer)
        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outLayer)
        nn.addConnection(in_to_hidden)
        nn.addConnection(hidden_to_out)
        nn.sortModules()
        nn.randomize()
        self.net_settings = str(nn.connections)
        self.nn = nn

    def buildIris(self):
        self.params['dataset'] = 'iris'
        self.trn_data, self.tst_data = pybrainData(0.5)
        global trn_data
        trn_data = self.trn_data
        nn = FeedForwardNetwork()
        inLayer = TanhLayer(4, name='in')
        hiddenLayer = TanhLayer(6, name='hidden0')
        outLayer = ThresholdLayer(3, name='out')
        nn.addInputModule(inLayer)
        nn.addModule(hiddenLayer)
        nn.addOutputModule(outLayer)
        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outLayer)
        nn.addConnection(in_to_hidden)
        nn.addConnection(hidden_to_out)
        nn.sortModules()
        nn.randomize()
        self.net_settings = str(nn.connections)
        self.nn = nn

trn_data = None
tst_data = None
exp_dir_str = None


def testOnTrainData(nn):
    outs = []
    for i, t in trn_data:
        outs.append(nn.activate(i))
    outs, trgs = OutToClass(outs, trn_data['target'])
    print outs
    print trn_data['class']
    return percentError(outs, trn_data['class'])



def runTrial(ds, iters):
    kw = {'minimize':True,  
        'verbose':True, 'mutationProb':0.1, 'populationSize':150,
        'storeAllEvaluations':True, 'storeAllPopulations':True,
        'elitism':True, 'tournament':False}
    e = setUpExperiment(ds, **kw)
    ga = GA(testOnTrainData, e.nn, **kw)
    e.results['trn_per_err'] = []
    for i in range(iters):
        print ga.learn(0)
        e.results['trn_per_err'].append(ga._bestFound()[1])
        fits = array(ga.fitnesses)
        print '\t\t\tmean', mean(fits), 'std', std(fits), 'max', max(fits), 'min', min(fits)
        for i, t in trn_data:
            o = ga._bestFound()[0].activate(i)
            if where(o != t, True, False).any():
                print i, o, argmax(t), '----'
    e.results['tst_per_err'].append(testOnTestData(ga._bestFound()[0], e.tst_data))
    return e

def testOnTestData(nn, tst_data):
    trgs = []
    outs = []
    for i, t in tst_data:
        outs.append(nn.activate(i))
        trgs.append(t)
    outs, trgs = OutToClass(outs, tst_data['target'])
    # print outs
    # print tst_data['class']
    return percentError(outs, tst_data['class'])    



def main ():
    ds, trials, iters = getArgs()
    all_trials = {'trn_per_err':[], 'tst_per_err':[]}
    for i in range(trials):
        trial = runTrial(ds, iters)
        for k in trial.results.keys():
            if trial.results[k]:
                all_trials[k].append(trial.results[k])
    writeResultsData(all_trials)
    writeExperimentParams(trial.params, trials, iters)


def setUpExperiment(ds, **kw):
    e = Experiment(**kw)
    if ds == 'xor':
        e.buildXor()
    elif ds == 'parity':
        e.buildParity()
    elif ds == 'iris':
        e.buildIris()
    return e

def preTrainWeights(e):
    trainer = bp(e.nn, dataset=trn_data, verbose=True)
    trainer.trainEpochs(100)    


def trainWithGA(e, ga, verbose=True):
    ga.learn(0)
    e.results['trn_per_err'].append(ga._bestFound()[1])
    fits = array(ga.fitnesses)
    if verbose:
        print '\t\t\tmean', mean(fits), 'std', std(fits), 'max', max(fits), 'min', min(fits)
    for i, t in trn_data:
        o = ga._bestFound()[0].activate(i)
        if where(o != t, True, False).any():
            print i, o, argmax(t), '----'
            

def writeExperimentParams(params, trials, iters):
    exp_params = params
    exp_params['trials'] = trials
    exp_params['iters'] = iters
    exp_params['trn_data_len_indim_outdim'] = (len(trn_data['input']), trn_data.indim, trn_data.outdim)
    if tst_data:
        exp_params['trn_data_shape'] = trn_data.shape
    f = open(exp_dir_str + 'implementation_parameters.out', 'w')
    f.writelines(str(params.items()))
    f.close()


def getArgs():
    len_args = len(sys.argv)
    ds = sys.argv[1] if len_args > 1 else 'xor'
    trials = int(sys.argv[2]) if len_args > 2 else 1
    iters = int(sys.argv[3]) if len_args > 3 else 4
    return ds, trials, iters


def makeTimeString():
    a = datetime.datetime.now().utctimetuple()
    time_string = str(a[3]) + str(a[4]) + '_' + str(a[2]) + '-' + \
        str(a[1]) + '-' + str(a[0])
    return time_string


def getOSSeparator():
    if os.environ['OS'] == 'Windows_NT':
        sep = '\\'
    else:
        sep = '/'
    return sep


def makeExpDir():
    time_string = makeTimeString()
    sep = getOSSeparator()
    cur_dir = os.getcwd() + sep
    os.mkdir(cur_dir + 'results' + sep + time_string)
    global exp_dir_str
    exp_dir_str = cur_dir + 'results' + sep + time_string + sep
    return exp_dir_str 


def writeResultsData(all_trials):
    exp_dir_str = makeExpDir()
    for k in all_trials.keys():
        if all_trials[k]:
            print array(all_trials[k]).shape 
            print exp_dir_str, k
            f = open(exp_dir_str + k + '.out', 'w')
            np.savetxt(f, all_trials[k])


if __name__ == '__main__':
    main()      