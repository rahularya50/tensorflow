# coding=utf-8
import math

import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from scipy.stats import linregress

import web_data

LOG_SPREAD = False
BLOCK_SIZE = 200


def gen_log_raws(raws):
	out = {}
	for stock, raw in raws.items():
		out[stock] = [math.log(val) for val in raw if val != 0]
	return out


def cointegration(seq1, seq2):
	if not (seq1 and seq2):
		return 0
	seq1, seq2 = seq1[-len(seq2):], seq2[-len(seq1):]
	if len(seq1) < 20:
		return 0
	return ts.coint(seq1, seq2)[1]


def find_best_pairs(raws, top=100):
	market_caps = web_data.get_market_caps()
	stocks = sorted(raws.keys(), key=lambda x: market_caps.get(x, 0), reverse=True)[:top]
	print(stocks)
	out = []
	for i in range(len(stocks)):
		print(i)
		for j in range(i+1, len(stocks)):
			out.append((cointegration(raws[stocks[i]], raws[stocks[j]]), stocks[i], stocks[j]))
	return out


def plot_pairs(s1, s2):
	seq1, seq2 = web_data.load_stocks()[s1], web_data.load_stocks()[s2]
	seq1, seq2 = seq1[-len(seq2):], seq2[-len(seq1):]

	slope, intercept, rvalue, pvalue, stderr = linregress(seq1, seq2)

	fig, ax = plt.subplots()
	ax.plot(seq1)
	ax.plot(seq2)
	ax.plot([j - slope * i - intercept for i, j in zip(seq1, seq2)])


def make_spreads(*, train):
	pairs = []
	spreads = []
	with open("target_pairs.csv") as file:
		for line in file:
			pairs.append(line.split())

	for s1, s2 in pairs:
		seq1, seq2 = web_data.load_stocks()[s1], web_data.load_stocks()[s2]
		seq1, seq2 = seq1[-len(seq2):], seq2[-len(seq1):]

		if LOG_SPREAD:
			seq1, seq2 = ([math.log(i) for i in seq if i != 0] for seq in (seq1, seq2))

		temp = blockify_spread((seq1, seq2))

		if train:
			temp = temp[:-math.ceil(len(temp) / 5)]
		else:
			temp = temp[-math.ceil(len(temp) / 5):]

		spreads.append(temp)

	return spreads


def blockify_spread(raw_spreads):
	spread = []
	for i in range(len(raw_spreads[0]) // BLOCK_SIZE - 1):
		prev_seq = [raw_spread[i*BLOCK_SIZE: (i+1)*BLOCK_SIZE] for raw_spread in raw_spreads]
		next_seq = [raw_spread[(i+1)*BLOCK_SIZE: (i+2)*BLOCK_SIZE] for raw_spread in raw_spreads]
		slope, intercept, rvalue, pvalue, stderr = linregress(prev_seq)
		print(slope, intercept)
		spread.append([j - slope*i - intercept for i, j in zip(*next_seq)])
	return spread


def plot_spread(i):
	pairs = []
	with open("target_pairs.csv") as file:
		for line in file:
			pairs.append(line.split())

	s1, s2 = pairs[i]
	seq1, seq2 = web_data.load_stocks()[s1], web_data.load_stocks()[s2]
	seq1, seq2 = seq1[-len(seq2):], seq2[-len(seq1):]

	if LOG_SPREAD:
		seq1, seq2 = ([math.log(i) for i in seq if i != 0] for seq in (seq1, seq2))

	spread = sum(make_spreads(train=True)[i] + make_spreads(train=False)[i], [])

	print(s1, s2, spread)

	fig, ax = plt.subplots()
	ax.plot(seq1)
	ax.plot(seq2)
	ax.plot(spread)


def gen_best_pairs():
	if LOG_SPREAD:
		pair_cointegrations = find_best_pairs(gen_log_raws(web_data.load_stocks()))
	else:
		pair_cointegrations = find_best_pairs(web_data.load_stocks())

	count = 0
	for coeff, s1, s2 in sorted(pair_cointegrations)[:50]:
		seq1, seq2 = web_data.load_stocks()[s1], web_data.load_stocks()[s2]
		seq1, seq2 = seq1[-len(seq2):], seq2[-len(seq1):]

		if 0.2 < linregress(seq1, seq2).slope < 5:
			print(s1, s2)
			plot_pairs(s1, s2)
			count += 1
		if count > 10:
			break


if __name__ == "main":
	pass
