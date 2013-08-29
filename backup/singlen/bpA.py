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

def showresults(errors, w, weights1, weights2):
  for p in pat:
    x = p[0]
    t = p[1]
    print x,'-->',tanh(sum([w[i]*x[i] for i in range(len(x))])),t  
  subplot(211)
  plot(errors,'rx')
  subplot(212)
  plot(weights1, 'g-')
  plot(weights2, 'b-')
  show()

def main():
  errors = []
  n = 0.6
  w = [r()]*len(pat[0][0])
  weights1, weights2 = [],[]
  Athresh = 3
  Adur = 3
  m_iters = 6
  for z in range(1000):
    error = 0
    Aa = 0
    Ac = 0
    weights1.append(w[0])
    weights2.append(w[1])
    for p in pat:
      x = p[0]
      t = p[1]
      
      for m in range(m_iters):
        lsum = sum([ w[i]*x[i] for i in range(len(x)) ])
        y = tanh(lsum)
        # astrocyte
        if y > 0: Aa +=1
        else: Aa -=1
        if Aa == Athresh:
          Ac = Adur
          Aa = 0
        if Aa == -Athresh:
          Ac = -Adur
          Aa = 0
        if Ac > 0:
          for i in range(len(x)):
            w[i] += w[i]*0.25
          Ac -=1
        if Ac < 0:
          for i in range(len(x)):
            w[i] += w[i]*-0.5
          Ac +=1
    weights1.append(w[0])
    weights2.append(w[1])
    for p in pat:
      x = p[0]
      t = p[1]
      lsum = sum([ w[i]*x[i] for i in range(len(x)) ])
      y = tanh(lsum)
      error += 0.5*(y-t)**2
      delta = (y-t)*sechsq(lsum)
      for i in range(len(x)):
        w[i] += delta*x[i]*n
    weights1.append(w[0])
    weights2.append(w[1])
    errors.append(error)
  showresults(errors,w, weights1, weights2)







if __name__ == "__main__":
    main()
