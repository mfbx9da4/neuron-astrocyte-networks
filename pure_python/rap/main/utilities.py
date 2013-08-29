from pybrain.datasets.classification import ClassificationDataSet
from numpy import empty, where, array, bincount, argmax
from numpy.random import permutation, rand
from collections import OrderedDict

def sumTo(num):
	if num: return num + sumTo(num-1)
	else:	return num


def xorDataSet():
    d = ClassificationDataSet(2)
    d.addSample([0., 0.], [0.])
    d.addSample([0., 1.], [1.])
    d.addSample([1., 0.], [1.])
    d.addSample([1., 1.], [0.])
    d.setField('class', [[0.], [1.], [1.], [0.]])
    return d


def randomConfig(ni, nh):
	return permutation(ni * nh)


def identifierMatrix(rows, cols):
	m = empty((rows, cols))
	for i in range(rows):
		for j in range(cols):
			m[i][j] = i + (float(j) / 10)
	return m


def initInterAstroWs(astrodim):
	weights = OrderedDict()
	for i in range(astrodim):
		k = i + 1
		for j in range(k, astrodim):
			weights[(i, j)] = rand()
	return weights


def convertToIndices(config, hiddim):
    indices_config = []
    for subset in config:
        astro_config = []
        for syn in subset:
            astro_config.append((syn / hiddim, syn % hiddim))
        indices_config.append(tuple(astro_config))
    return indices_config

def OutToClass(outputs, targets):
    outs = []
    trgs = []
    for out, trg in zip(outputs, targets):
        # this is to have each output neruon responsible for a class
        # if False and len(out) == 3 and \
        if len(out) == 1:
          outs.append(1. if out[0] >= 0.5 else 0.)
          trgs.append(trg[0])
          # print out[0]  
        # elif len(out) == 3 and \
        #   array([where(x in [1, 0], True, False) for x in out]).all() and \
        #   bincount(out.astype('int'))[0] != 2:
        #     outs.append(-1)
        #     trgs.append(argmax(trg))
        else:
            outs.append(argmax(out))
            trgs.append(argmax(trg))
    return outs, trgs

di = OrderedDict({'a':5, 'b':4, 'd':4})