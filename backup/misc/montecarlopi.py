from decimal import *
from random import random
from math import pow, sqrt

DARTS=1e7
hits = 0
throws = 0
for i in range (1, int(DARTS)):
	throws += 1
	x = random()
	y = random()
	distance_from_origin = sqrt(pow(x, 2) + pow(y, 2))
	if distance_from_origin <= 1.0:
		hits = hits + 1.0

# hits / throws = 1/4 Pi
pi = Decimal(4 * (hits / throws))

print "pi = %s" %(pi)
