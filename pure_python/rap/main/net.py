from utilities import convertToIndices, sumTo
from utilities import initInterAstroWs
from scipy import empty, zeros, tanh, concatenate, split, cumsum, array
from numpy.random import rand, permutation
from scipy import sum as npsum
from astro import Astro


class NeuroAstroNet(object):
    def __init__(self, indim, hiddim, outdim, **kw):
        """
        syn_wi = synapse-to-astrocyte weights
        syn_wo = astro-to-synapse weights
        inter_ws = inter-astrocyte weights
        wi = input weights (neuronal)
        """
        self.initSettings(**kw)
        self.initDimensions(indim, hiddim, outdim)
        self.initNeurons()
        self.initWeights()
        self.initAstrocytes()

    def initSettings(self, **kw):
        self.momentum = kw['momentum'] if kw.has_key('momentum') else 0.2
        self.lag_iters = kw['lag_iters'] if kw.has_key('lag_iters') else 2

    def initDimensions(self, indim, hiddim, outdim):
        self.indim = indim
        self.hiddim = hiddim
        self.astrodim = hiddim # TBC
        self.outdim = outdim

    def initNeurons(self):
        self.ai = empty(self.indim)
        self.ah = empty(self.hiddim)
        self.ao = empty(self.outdim)

    def initWeights(self):
        ni = self.indim
        nh = self.hiddim
        no = self.outdim
        self.wi = -2 + rand(ni, nh) * 4
        self.wo = -2 + rand(nh, no) * 4
        self.syn_wi = -0.05 + rand(ni, nh) * 0.1  # initialize to very low values
        self._inter_ws = initInterAstroWs(self.astrodim)
        self.syn_wo = -2 + rand(ni, nh) * 4
        self.shapes = {
        'wi' : self.wi.shape, 
        'syn_wo' : self.syn_wo.shape, 
        'inter_ws' : len(self.inter_ws), 
        'syn_wi' : self.syn_wi.shape, 
        'wo' : self.wo.shape}

    @property
    def inter_ws(self):
        return array(self._inter_ws.values())

    @inter_ws.setter
    def inter_ws(self, values):
        for i, k in enumerate(self._inter_ws.iterkeys()):
            self._inter_ws[k] = values[i]

    def initAstrocytes(self):
        """config is numbered input synapses aranged in shape of 
        syns are tuples of non-overlapping synapses"""
        sf = self
        config = permutation(sf.indim * sf.hiddim).reshape(sf.astrodim, sf.indim)
        config = convertToIndices(config, sf.hiddim)
        self.astros = [Astro(syns, self.lag_iters) for syns in config]
        self.astroOuts = empty((self.indim, self.hiddim))

    def activate(self, inputs):
        self.activateIn(inputs)
        self.activateAstros()
        self.activateHid()
        self.activateOut()
        return self.ao.copy()

    def activateIn(self, inputs):
        assert len(inputs) == self.indim, 'incorrect number of inputs'
        self.ai[:] = tanh(inputs)

    def activateHid(self):
        self.ah[:] = tanh(sum(self.wi.T * self.ai, axis=1))

    def activateOut(self):
        self.ao[:] = tanh(npsum(self.wo.T * self.ah, axis=1))

    def activateAstros(self):
        self.updateFromSynapse()
        self.updateFromLag()
        self.updateFromOtherAstros()
        out = self.astroOut()
        self.storeLag()
        self.resetAstroActs()
        return out

    def updateFromSynapse(self):
        for astro in self.astros:
            for i, j in astro.syns:
                astro.act +=  self.syn_wo[i][j] * self.ai[i]

    def updateFromLag(self):
        for astro in self.astros:
            astro.act += float(astro.prev * self.momentum)

    def updateFromOtherAstros(self):
        temp = zeros(self.astrodim)
        for x, y in self._inter_ws:
             temp[x] += self._inter_ws[(x, y)] * self.astros[y].act
             temp[y] += self._inter_ws[(x, y)] * self.astros[x].act
        for i, astro in enumerate(self.astros):
            astro.act += temp[i]

    def astroOut(self):
        for astro in self.astros:
            astro.act = tanh(astro.act)
            for syn in astro.syns:
                i, j = syn
                self.astroOuts[i][j] = astro.act * self.syn_wi[i][j]  
        return self.astroOuts.copy()

    def storeLag(self):
        for i, astro in enumerate(self.astros):
            self.astros[i].prev = astro.act

    @property
    def weights(self):
        """in order of weight activations"""
        syn_wo = self.syn_wo.flatten()
        inter_ws = self.inter_ws.copy()
        syn_wi = self.syn_wi.flatten()
        wi = self.wi.flatten()
        wo = self.wo.flatten()
        weights = concatenate((syn_wo, inter_ws, syn_wi, wi, wo))
        return weights
    
    @weights.setter
    def weights(self, new_weights):
        sections = shapesToSections(self.shapes)
        syn_wo, inter_ws, syn_wi, wi, wo = split(new_weights.copy(), sections)
        shapes = self.shapes
        self.syn_wo = syn_wo.reshape(shapes['syn_wo'])
        self.inter_ws = inter_ws
        self.syn_wi = syn_wi.reshape(shapes['syn_wi'])
        self.wi = wi.reshape(shapes['wi'])
        self.wo = wo.reshape(shapes['wo'])

    def resetAstros(self):
        for astro in self.astros:
            astro.act = 0.
            astro._prev = self._prev = [0. for i in range(self.lag_iters)]

    def resetAstroActs(self):
        for astro in self.astros:
            astro.act = 0.


def shapesToSections(shapes):
    sections = []
    sections.append(shapes['syn_wo'][0] * shapes['syn_wo'][1])
    sections.append(shapes['inter_ws'][0] * shapes['inter_ws'][1])
    sections.append(shapes['syn_wi'][0] * shapes['syn_wi'][1])
    sections.append(shapes['wi'][0] * shapes['wi'][1])
    sections.append(shapes['wo'][0] * shapes['wo'][1])
    return cumsum(sections)
