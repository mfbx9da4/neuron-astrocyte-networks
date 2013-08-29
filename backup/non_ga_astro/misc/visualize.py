import networkx as nx
from pylab import *

class BuildGraph(object):
	def __init__(self, *dims):
		self.g = nx.Graph()
		self.dims = dims
		self.addNodes()
		layers = self.getLayers()
		positions = []
		print layers
		for l in layers:
			positions += [n['pos'] for n in l]
		print self.g.nodes()
		print array(positions).ravel()
		posits =  dict(zip(self.g.nodes(), array(positions).flatten()))
		nx.draw_networkx_nodes(self.g, posits)
		show()

	def addNodes(self):
		mx_dim = max(self.dims)
		prefix = ['i', 'j', 'k']
		for pre, dim in zip(prefix, self.dims):
			offset = (0.5 * (mx_dim - dim))
			positions = arange(dim) + offset
			for i in range(dim):
				name = pre + str(i)
				self.g.add_node(name)
				self.g[name]['pos'] = positions[i]
				self.g[name]['act'] = 0.0
				print self.g[name]

	def getLayers(self):
		indim, hiddim, outdim = self.dims
		in_l = [self.g['i' + str(x)] for x in range(indim)]
		hid_l = [self.g['j' + str(x)] for x in range(hiddim)]
		out_l = [self.g['k' + str(x)] for x in range(outdim)]
		all_ls = [in_l, hid_l, out_l]
		return all_ls

def main():
	BuildGraph(2, 3, 1)

if __name__ == '__main__':
	main()
