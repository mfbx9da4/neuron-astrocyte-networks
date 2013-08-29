from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import FeedForwardNetwork, RecurrentNetwork
n= FeedForwardNetwork()

# layers can be named
inLayer = LinearLayer(2, name='foo')
hiddenLayer = SigmoidLayer(3)
outLayer = LinearLayer(1)

# add each layer, connect them individually and add the connections
# to the MLP
n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)
n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)

# this is required to make the MLP usable 
n.sortModules()

print n.activate((2,2)) # forward pass
print 'n.params\n', n.params # all weights

# same but for recurrent network
n = RecurrentNetwork()
n.addInputModule(LinearLayer(2, name='in'))
n.addModule(SigmoidLayer(3, name='hidden'))
n.addOutputModule(LinearLayer(1, name='out'))
n.addConnection(FullConnection(n['in'], n['hidden'], name='c1'))
n.addConnection(FullConnection(n['hidden'], n['out'], name='c2'))
n.addRecurrentConnection(FullConnection(n['hidden'], n['hidden'], name='c3'))

n.sortModules()


print n.activate((2,2)) # forward pass
print n.activate((2,2)) # forward pass
print n.activate((2,2)) # forward pass
print n.reset(), '\nafter reset'
print n.activate((2,2)) # forward pass
