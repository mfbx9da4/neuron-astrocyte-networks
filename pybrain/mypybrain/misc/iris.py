# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 19:29:02 2013

@author: david
"""
import urllib2

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

s = urllib2.urlopen(url)
s.