from pylab import *
import time

figure()
show()

ion()
p, = plot(arange(1000) * -1)
p, = plot(arange(1000) * 1)
for i in linspace(-1, 1, 100):
	# time.sleep(1.)
	p.set_ydata(arange(1000)*i)
	i += 1
	draw()

