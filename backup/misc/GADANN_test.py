import GADANN as g
from ANGN import pop_size
  
  #passed = ''
  #if passed != '': print passed


def newpopWchanged(newpopW, copy):
  passed = ''
  for i in range(len(copy)):
    if newpopW[i] != copy[i]:
      passed+= str(i)
    
  
  if passed != '': print 'newpopW IS CHANGED AFTER BEING USED FOR ASSIGNMENT',passed

def correctlyranked (rankedPop):
  passed = ''
  for i in range(pop_size-3):
    if not rankedPop[i][1] > rankedPop[i+1][1]:
      passed += str(i)
  
  if passed != '': print passed


def dezip (rankedPop, weights, fitnesses):
  passed = ''
  if weights != [ item[0] for item in rankedPop ]:
    passed += 'RANKED WEIGHTS COPIED INCORRECTLY\n'
  if fitnesses != [ item[-1] for item in rankedPop ]:
    passed += 'FITNESS SCORES COPIED INCORRECTLY\n'
  if passed != '': print passed

def elitism(rankedWeights, newpopW):
  passed = ''
  x = int(pop_size*0.15)
  if newpopW[:x] != rankedWeights[:x]:
    passed+= 'ELITISM INCORRECT '+ str(len(newpopW))
  for i in range(x):
    if rankedWeights[i] != newpopW[i]:
      passed += str(i)
  
  if passed != '': print passed


