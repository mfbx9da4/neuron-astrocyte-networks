from operator import itemgetter, attrgetter
import random
import sys
import os


# GLOBAL VARIABLES

genetic_code = {
  '0000':'0',
  '0001':'1',
  '0010':'2',
  '0011':'3',
  '0100':'4',
  '0101':'5',
  '0110':'6',
  '0111':'7',
  '1000':'8',
  '1001':'9',
  '1010':'+',
  '1011':'-',
  '1100':'*',
  '1101':'/'
  }

#  when editing the variable us the loabal keyword e,g.  global solution_found
solution_found = False
popN = 100 # n number of chromos per population
genesPerCh = 75
max_iterations = 10
target = 100
crossover_rate = 1
mutation_rate = 1

def generatePop ():
  chromos, chromo = [], []
  for eachChromo in range(popN):
    chromo = []
    for bit in range(genesPerCh * 4):
      chromo.append(random.randint(0,1))
    chromos.append(chromo)
  return chromos

def translate (chromo):
  protein, chromo_string = '',''
  need_int = True
  a, b = 0, 4 # ie from point a to point b (start to stop point in string)
  for bit in chromo:
    chromo_string += str(bit)
  #print chromo_string[a:b]
  for gene in range(genesPerCh):
    if chromo_string[a:b] == '1111' or chromo_string[a:b] == '1110':
      continue
    elif chromo_string[a:b] != '1010' and chromo_string[a:b] != '1011' and chromo_string[a:b] != '1100' and chromo_string[a:b] != '1101':
      if need_int == True:
        protein += genetic_code[chromo_string[a:b]]
        need_int = False
        a += 4
        b += 4
        continue
      else:
        a += 4
        b += 4
        continue
    else:
      if need_int == False:
        protein += genetic_code[chromo_string[a:b]]
        need_int = True
        a += 4
        b += 4
        continue
      else:
        a += 4
        b += 4
        continue
  if len(protein) %2 == 0:
    protein = protein[:-1]
  return protein
  

def evaluate(protein):
  a = 3
  b = 5
  output = -1
  lenprotein = len(protein) # i imagine this is quicker than calling len everytime?
  if lenprotein == 0:
    output = 0
  if lenprotein == 1:
    output = int(protein)
  if lenprotein >= 3:
    try :
      output = eval(protein[0:3])
    except ZeroDivisionError:
      output = 0
    if lenprotein > 4:
      while b != lenprotein+2:
        try :
          output = eval(str(output)+protein[a:b])
        except ZeroDivisionError:
          output = 0
        a+=2
        b+=2  
  return output

def calcFitness (errors):
  fitnessScores = []
  totalError = sum(errors)
  i = 0
  for error in errors:
    fitnessScores.append (float(errors[i])/float(totalError))
    i += 1
  return fitnessScores

def rankPop (chromos):
  proteins, outputs, errors = [], [], []
  i = 1
  for chromo in chromos:
    protein = translate(chromo)
    proteins.append(protein)
    
    output = evaluate(protein)
    outputs.append(output)
    
    try:
      error = float(1)/(float(abs(target-output)))
    except ZeroDivisionError:
      global solution_found
      solution_found = True
      error = 0
      print 'Solution =', protein, '=', output
      break
    else:
      error = float(1)/(float(abs(target-output)))
      errors.append(error)
    print '(',i,')', protein, '=', output
    i+=1  
  fitnessScores = calcFitness (errors)
  pairedPop = zip ( chromos, proteins, outputs, fitnessScores)
  rankedPop = sorted ( pairedPop, key = itemgetter(-1) )
  return rankedPop
  
def selectFittest (rankedPop):
  fitnessScores = []
  fitnessScores = [ item[-1] for item in rankedPop ]
  rankedChromos = [ item[0] for item in rankedPop ]
  index1 = roulette (fitnessScores)
  index2 = roulette (fitnessScores)
  ch1 = rankedChromos[index1]
  ch2 = rankedChromos[index2]
  return ch1, ch2
  
def roulette (fitnessScores):
  index = 0
  cumalativeFitness = 0
  cumalativeFitness = float(cumalativeFitness)
  for fitness in fitnessScores:
    if random.random() < cumalativeFitness:
      return index
    cumalativeFitness += float(fitness)
    index += 1

def crossover (ch1, ch2):
  r = random.randint(0,genesPerCh*4)
  ch1 = ch1[:r]+ch2[r:]
  ch2 = ch2[:r]+ch1[r:]
  return ch1, ch2

def mutate (ch):
  for i in ch:
    if random.random() < mutation_rate:
      if ch[i] == 1:
        ch[i] = 0
      else:
        ch[i] = 1
  return ch
      

def breed (ch1, ch2):
  if random.random() < crossover_rate:
    ch1, ch2 = crossover(ch1, ch2)
  ch1 = mutate (ch1)
  ch2 = mutate (ch2)
  return ch1, ch2

def iteratePop (rankedPop):
  chromos = []
  while len(chromos) != popN:
    ch1, ch2 = selectFittest (rankedPop)
    ch1, ch2 = breed (ch1, ch2)
    chromos.append(ch1)
    chromos.append(ch2)
  return chromos
      
def main():
  chromos = generatePop()
  iterations = 0
  while iterations != max_iterations or solution_found != True:
    print '\nCurrent iterations:', iterations
    rankedPop = rankPop(chromos)
    if solution_found != True:
      chromos = []
      chromos = iteratePop(rankedPop)
      i = 0
      for x in chromos:
        i += 1
      print i
      iterations += 1
    else:
      break

  
  
  
  
    
if __name__ == "__main__":
    main()
