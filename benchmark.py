# coding=utf-8

import numpy

THRESHOLD = 1.5
LEN_HISTORY = 20


def benchmark_buy(history, threshold):
	if len(history) < LEN_HISTORY:
		return False
	stdev = numpy.std(history[-LEN_HISTORY:])
	return history[-1] < - threshold * stdev


def benchmark_sell(history, threshold):
	if len(history) < LEN_HISTORY:
		return False
	stdev = numpy.std(history[-LEN_HISTORY:])
	return history[-1] > threshold * stdev


def gen_predictors(threshold_min=THRESHOLD, threshold_max=THRESHOLD):
	return lambda history: benchmark_buy(history, threshold_min),\
	       lambda history: benchmark_sell(history, threshold_max),\
	       "benchmark"
