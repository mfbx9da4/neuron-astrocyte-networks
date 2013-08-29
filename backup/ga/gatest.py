import random

def crossover (ch1, ch2):
  r = random.randint(0,5)
  newCh1 = ch1[:r]+ch2[r:]
  newCh2 = ch2[:r]+ch1[r:]
  #print 'newCh1', newCh1
  #print 'newCh2', newCh2, '\n'
  #assert newCh1 != ch1
  return newCh1, newCh2

def mutate (ch):
  mutatedCh = []
  for i in ch:
    if random.random() < 1:
      if i == 1:
        mutatedCh.append(11)
      else:
        mutatedCh.append(22)
    else:
      mutatedCh = ch
  assert mutatedCh != ch
  return mutatedCh
      

def breed (ch1, ch2):
  #assert ch1 != ch2 
  print 'ch1', ch1
  print 'ch2', ch2, '\n'
  newCh1, newCh2 = [], []
  if random.random() < 1:
    newCh1, newCh2 = crossover(ch1, ch2)
  newnewCh1 = mutate (newCh1)
  newnewCh2 = mutate (newCh2)
  #assert newnewCh1 != ch1
  #assert newnewCh2 != ch2
  print 'newnewCh1', newnewCh1
  print 'newnewCh2', newnewCh2, '\n'
  return newnewCh1, newnewCh2

def main ():
  chromos = []
  ch1 = [1, 1, 1, 1, 1]
  ch2 = [2, 2, 2, 2, 2]
  while len(chromos) != 10:
    #ch1, ch2 = [], []
    #ch1, ch2 = selectFittest (rankedPop)
    newch1, newch2 = breed (ch1, ch2)
    chromos.append(newch1)
    chromos.append(newch2)
  #for x in chromos:
   # print chromos
  return chromos


if __name__ == "__main__":
    main()
