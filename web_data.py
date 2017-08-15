# coding=utf-8

import pandas_datareader.data as web
import datetime as dt
from scipy.stats import linregress

BLOCK_SIZE = 100


def gen_spread(*stocks, train=False):
	start = dt.datetime(2008, 7, 26)
	end = dt.datetime(2017, 7, 26)

	dfs = [web.DataReader(stock, "google", start, end) for stock in stocks]
	raws = [list(df["Close"]) for df in dfs]

	if len(raws) != 2:
		raise TypeError("gen_spread() takes exactly 2 arguments")

	if train:
		raw_spreads = [raw[:-1 * len(raws[0]) // 5] for raw in raws]
	else:
		raw_spreads = [raw[-1 * len(raws[0]) // 5:] for raw in raws]

	spread = []
	for i in range(len(raw_spreads[0]) // BLOCK_SIZE - 1):
		prev_seq = [raw_spread[i*BLOCK_SIZE: (i+1)*BLOCK_SIZE] for raw_spread in raw_spreads]
		next_seq = [raw_spread[(i+1)*BLOCK_SIZE: (i+2)*BLOCK_SIZE] for raw_spread in raw_spreads]
		slope, intercept, rvalue, pvalue, stderr = linregress(prev_seq)
		print(slope, intercept)
		spread.append([j - slope*i - intercept for i, j in zip(*next_seq)])

	return spread
