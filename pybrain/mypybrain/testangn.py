# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 19:50:43 2013

@author: david
"""

import unittest
from angn import *
from numpy import zeros, ones

class testData(unittest.TestCase):
    def setUp(self):
        pass
    
    def testNumberOfClasses(self):
        for i in range(3):
            self.assertEqual(list(alldata['target']).count(i), 50)
    def testNumberOfTrainingPatterns(self):
        self.assertEqual(len(trndata), 101)
    def testNumberOfTestPatterns(self):
        self.assertEqual(len(tstdata), 49)
    def testDimesnsions(self):
        self.assertEqual(trndata.indim, 4)
        self.assertEqual(tstdata.indim, 4)
        self.assertEqual(trndata.outdim, 3)
        self.assertEqual(tstdata.outdim, 3)
    def testAllSamples(self):
        # use urllib2 to request iris.data
        # or protect iris.data
        r = random.randint(0, len(trndata))
        print "Sample %d (input, target, class):" % r
        print trndata['input'][r], trndata['target'][r], trndata['class'][r], \
          trndata.getClass(int(trndata['class'][r][0]))
      
class testAstrocyteCopyUpdatedByBP(unittest.TestCase):
    def setUP(self):
        pass

  
    def tearDown(self):
        pass

  
    


if __name__== '__main__':
  unittest.main()
