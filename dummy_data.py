# coding=utf-8

import numpy

MEAN = 0
DURATION = 500
STDEV = 1
DECAY = 0.9


def gen_sim_data(length, mean, rho, sigma):
	out = [mean]
	for i in range(length - 2):
		out.append(rho * (out[-1] - mean) + mean + gaussian(sigma))
	return out


def gaussian(sigma):
	return float(numpy.random.normal(0, sigma))


