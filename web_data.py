# coding=utf-8

import datetime as dt
import pickle

import pandas
import pandas_datareader.data as web
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


def download_stocks(start_index=0):
	url_nasdaq = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"

	df = pandas.DataFrame.from_csv(url_nasdaq)
	stocks = df.index.tolist()[start_index:]

	start = dt.datetime(2000, 7, 26)
	end = dt.datetime(2017, 7, 26)

	raws = {}

	for i, stock in enumerate(stocks):
		try:
			raws[stock] = list(web.DataReader(stock, "google", start, end)["Close"])
			print("{} {} / {}".format(stock, i + start_index, len(stocks) + start_index))
		except Exception as e:
			print(e)
		if i % 100 == 0 and i > 0:
			with open("raws/raw_{}.pickle".format(i + start_index), "wb") as file:
				pickle.dump(raws, file)
			raws = {}

	with open("raws/raw_{}.pickle".format(len(stocks)), "wb") as file:
		pickle.dump(raws, file)


def get_market_caps():
	url_nasdaq = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"

	df = pandas.DataFrame.from_csv(url_nasdaq)
	caps = df["MarketCap"].to_dict()

	for stock in caps:
		if caps[stock] == "n/a":
			caps[stock] = 0
			continue
		temp = caps[stock][1:]
		temp = float(temp[:-1]) * {"M": 10**6, "B": 10**9}.get(temp[-1], 0)
		caps[stock] = temp

	return caps


def load_stocks():
	raws = {}
	for i in range(100, 3300, 100):
		with open("raws/raw_{}.pickle".format(i), "rb") as file:
			raws.update(pickle.load(file))
	return raws
