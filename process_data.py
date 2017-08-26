# coding=utf-8
import math

import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from scipy.stats import linregress

import web_data


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
	fig, ax = plt.subplots()
	seq1, seq2 = web_data.load_stocks()[s1], web_data.load_stocks()[s2]
	seq1, seq2 = seq1[-len(seq2):], seq2[-len(seq1):]

	slope, intercept, rvalue, pvalue, stderr = linregress(seq1, seq2)

	ax.plot(seq1)
	ax.plot(seq2)
	ax.plot([j - slope * i - intercept for i, j in zip(seq1, seq2)])


if __name__ == "main":
	# x = find_best_pairs(gen_log_raws(web_data.load_stocks()))
	y = find_best_pairs(web_data.load_stocks())

	count = 0
	for coeff, s1, s2 in sorted(y)[:50]:
		seq1, seq2 = web_data.load_stocks()[s1], web_data.load_stocks()[s2]
		seq1, seq2 = seq1[-len(seq2):], seq2[-len(seq1):]

		if 0.2 < linregress(seq1, seq2).slope < 5:
			print(coeff, s1, s2)
			plot_pairs(s1, s2)
			count += 1
		if count > 10:
			break
