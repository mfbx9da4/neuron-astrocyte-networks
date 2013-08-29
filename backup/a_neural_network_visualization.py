from matplotlib import pylab
import random
import numpy


fig=pylab.figure(dpi=120)
ax=fig.add_subplot(111)
patches = []

net = [[1,2,3], [4,5,6,7,8,9], [10,11,12,15], [13,14]]
x_ys = []
neuron_colors = []
connection_colors=[]


"""
pretend data
"""
connection_strengths=[]
for layer in range(len(net)-1):
	connection_strengths.append([ numpy.random.uniform(-1,1,len(net[layer+1]))  for n in net[layer]])

print connection_strengths

"""
"""


#net must have at least 3 layers, (input, hidden, output)
horizontal_grid_size = len(net)
vertical_grid_size = 0
standard_radius = 10
standard_y_margin = 10
standard_x_margin = 100


"""
First Run through Draw
"""

for column in net:
	vertical_grid_size = max( vertical_grid_size,len(column) )


for layer in range(len(net)):

	starting_y = (vertical_grid_size / 2.0) - (len(net[layer]) / 2.0)

	x_ys.append([])
	neuron_colors.append([])

	#set up all default connections as black
	if layer+1<len(net):
		connection_colors.append([ ['k']*len(net[layer+1])  for n in net[layer]])


	for neuron in range(len(net[layer])):

		_x = standard_x_margin +( (layer) * standard_x_margin) + (standard_radius )

		_y = 100 + (starting_y+neuron)*(standard_radius+2*standard_y_margin) 

		x_ys[layer].append((_x,_y))


		#set up all neurons initially as white
		neuron_colors[layer].append('w')
		

		if layer < len(net)-1:

			starting_y2 = (vertical_grid_size / 2.0) - (len(net[layer+1]) / 2.0)

			
			for connection in range(len(net[layer+1])):


				__x = standard_x_margin +( (layer+1) * standard_x_margin) + (standard_radius )

				__y = 100 + (starting_y2+connection)*(standard_radius+2*standard_y_margin) 

				color = (connection_strengths[layer][neuron][connection]*8388607)+8388607

				connection_colors[layer][neuron][connection]="#%06.0x"%color

				pylab.plot( [_x, __x], [_y, __y], zorder = 1, color=connection_colors[layer][neuron][connection])



		pylab.plot([_x],[_y],'.',markersize=(5*standard_radius) , zorder=10, color=neuron_colors[layer][neuron])

		pylab.text(_x, _y, net[layer][neuron], color='k', zorder = 11, horizontalalignment='center', verticalalignment='center')


	layer += 1


ax.set_xbound(0, (2*standard_radius + standard_x_margin)*len(net))
ax.set_ybound(0,vertical_grid_size*50+50)

print connection_colors


pylab.show()	
