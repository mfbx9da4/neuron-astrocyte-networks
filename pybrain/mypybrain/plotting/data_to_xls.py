from datetime import date, time, datetime
from decimal import Decimal
import os
import numpy as np
import sys
from xlwt import Workbook, Style
import re
wb = Workbook()

def main():
	assert len(sys.argv) > 1
	args = sys.argv[1:]
	path = args[0]
	dir_name = re.findall('\d+_\d+.*', path)
	file_name = 'trn_per_err'
	try:
	    f = open(path + file_name + '.out', 'r')
	    trn_err = np.loadtxt(f)
	except IOError:
	    f = open(path + 'train_percents_top' + '.out', 'r')
	    trn_err = np.loadtxt(f)
	    for i, trial in enumerate(trn_err):
	        trn_err[i][:] = 100 - trn_err[i]
	file_name = 'tst_per_err'
	try:
	    f = open(path + file_name + '.out', 'r')
	    tst_err = np.loadtxt(f)
	except IOError:
	    f = open(path + 'test_percent' + '.out', 'r')
	    tst_err = np.loadtxt(f)
	    for i, trial in enumerate(tst_err):
	        tst_err[i] = 100 - tst_err[i]
	assert trn_err.shape == (20, 1000)
	assert tst_err.shape == (20, )
	ws = wb.add_sheet('All trials')
	ws.row(0).write(1, args[1])
	ws.row(1).write(0, 'Trials')
	ws.row(1).write(1, 'Training error')
	ws.row(1).write(2, 'Test error')
	for i, trn in enumerate(trn_err[:,-1]):
		ws.row(i+2).write(0, i)
		ws.row(i+2).write(1, trn)
		ws.row(i+2).write(2, tst_err[i])
	ws = wb.add_sheet('All Iters')
	ws.row(0).write(0, 'Iters')
	ws.row(0).write(1, args[1])
	for i, e in enumerate(np.mean(trn_err, axis=0)):
		ws.row(i+1).write(0, i)
		ws.row(i+1).write(1, e)




	


	wb.save(path + args[1] + '_data.xls')

if __name__ == '__main__':
	main()