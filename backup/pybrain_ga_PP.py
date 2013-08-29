import sys
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
import random
from scipy import std, mean, array, bincount, where, argmax
from pylab import plot, show

from myangn.sem6.iris import pybrainData
from mypybrain.astrocyte_layer import AstrocyteLayer
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
        self.results = {'train':[], 'test':[]}

    def buildXor(self):
        self.dataset = 'XOR'
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
        self.dataset = 'parity'
        self.trn_data = ParityDataSet(nsamples=75)
        self.trn_data.setField('class', self.trn_data['target'])
        self.tst_data = ParityDataSet()
        global trn_data
        trn_data = self.trn_data
        print trn_data
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
        self.dataset = 'iris'
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


def testOnTrainData(nn):
    outs = []
    for i, t in trn_data:
        outs.append(nn.activate(i))
    outs, trgs = OutToClass(outs, trn_data['target'])
    # print outs
    # print trn_data['class']
    return percentError(outs, trn_data['class'])


def runTrial(iters):
    kw = {'minimize':True,  
        'verbose':True, 'mutationProb':0.1, 'populationSize':150,
        'storeAllEvaluations':True, 'storeAllPopulations':True,
        'elitism':True, 'tournament':False}
    e = Experiment(**kw)
    # e.buildXor()
    e.buildParity()
    # e.buildIris()

    # trainer = bp(e.nn, dataset=trn_data, verbose=True)
    # trainer.trainEpochs(100)

    ga = GA(testOnTrainData, e.nn, **kw)
    # ga = GA(trn_data.evaluateModuleMSE, e.nn, **kw)
    # tmp_nn = e.nn.newSimilarInstance()
    perc_errs = []
    for i in range(iters):
        ga.learn(0)
        perc_errs.append(ga._bestFound()[1])
        fits = array(ga.fitnesses)
        print '\t\t\tmean', mean(fits), 'std', std(fits), 'max', max(fits), 'min', min(fits)
        for i, t in trn_data:
            o = ga._bestFound()[0].activate(i)
            if o != t:
                print i, o, t, '++' if o == t else '----'

        # ranked_pop = ga._allGenerations[i]
        # ranked_pop = array(sorted(ga._allGenerations[i], key=lambda x: x[1]))
        # for ind in range(len(ranked_pop[0])):
        #     tmp_nn.params[:] = array(ranked_pop[0][ind]).copy()
            # print ranked_pop[1][ind], testOnTrainData(tmp_nn), int(testOnTrainData(tmp_nn)) * '.'
            # astrocytes = associateAstrocyteLayers(tmp_nn)
            # trainNGA(tmp_nn, trn_data, astrocytes[0], astrocytes[1])

        # ga.currentpop = new_pop
    return perc_errs


def main ():
    if len(sys.argv) == 2:
        iters = int(sys.argv[1])
    else: 
        iters = 4
    runTrial(iters)
    # all_trials = {
    #     'train_mses_top' : [],
    #     'train_mses_avg' : [],
    #     'train_percents_avg' : [],
    #     'train_percents_top' : [],
    #     'test_mse' : [],
    #     'test_percent' : []
    # }
    # for i in range(trials):
    #     trial_results = runTrial()
    #     for k in trial_results.keys():
    #         all_trials[k].append(trial_results[k])

    # a = datetime.datetime.now().utctimetuple()
    # time_string = str(a[3]) + str(a[4]) + '_' + str(a[2]) + '-' + \
    #     str(a[1]) + '-' + str(a[0])
    # if os.environ['OS'] == 'Windows_NT':
    #     sep = '\\'
    # else:
    #     sep = '/'
    # cur_dir = os.getcwd() + sep
    # os.mkdir(cur_dir + time_string)
    # for k in all_trials.keys():
    #     f = open(cur_dir + time_string + sep + k + '.out', 'w')
    #     np.savetxt(f, all_trials[k])
    
    # implementation_parameters = {
    # 'trials' : trials,
    # 'max_iterations' : max_iterations,
    # 'pop_size' : NN.pop_size,
    # 'elitism_individuals_num' : NN.eliteN,
    # 'mutation_rate' : NN.mutation_rate,
    # 'crossover_rate' : NN.crossover_rate,
    # 'NN_indim' : NN.ni,
    # 'NN_hiddim' : NN.ni,
    # 'NN_outdim' : NN.no
    # }
    # f = open(cur_dir + time_string + sep + 
    #     'implementation_parameters.out', 'w')
    # f.writelines(str(implementation_parameters.items()))
    # f.close()


if __name__ == '__main__':
    main()      