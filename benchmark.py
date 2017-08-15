# coding=utf-8

THRESHOLD = 2


def benchmark_buy(history, threshold, avg=None):
	if avg is None and len(history) < 20:
		return False
	if avg is None:
		avg = sum(history[:20]) / 20
	return history[-1] < avg - threshold


def benchmark_sell(history, threshold, avg=None):
	if avg is None and len(history) < 20:
		return False
	if avg is None:
		avg = sum(history[:20]) / 20
	return history[-1] > avg + threshold


def gen_predictors(avg=None, threshold_min=THRESHOLD, threshold_max=THRESHOLD):
	return lambda history: benchmark_buy(history, threshold_min, avg),\
	       lambda history: benchmark_sell(history, threshold_max, avg),\
	       "benchmark"
