from random import random as r
from math import *
from matplotlib.pyplot import *
pat = [
  [[0,0], 0],
  [[0,1], 1],
  [[1,0], 1],
  [[1,1], 0]
]

def sechsq(x):
  return (4*e**(2*x))/(e**(4*x)+2*e**(2*x)+1)

def main():
  errors = []
  n = 0.6
  w = [r()]*len(pat[0][0])
  for z in range(5000):
    error = 0
    for p in pat:
      x = p[0]
      t = p[1]
      lsum = sum([ w[i]*x[i] for i in range(len(x)) ])
      y = tanh(lsum)
      error += 0.5*(y-t)**2
      delta = (y-t)*sechsq(lsum)
      for i in range(len(x)):
        w[i] += delta*x[i]*n
    errors.append(error)
  for p in pat:
    x = p[0]
    t = p[1]
    print x,'-->',tanh(sum([w[i]*x[i] for i in range(len(x))])),t  
  plot(errors,'rx')
  show()






if __name__ == "__main__":
    main()
