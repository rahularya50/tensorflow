# coding=utf-8
import tensorflow as tf
import tflearn
import numpy as np
import random
from tflearn.layers.core import input_data, fully_connected, reshape
from tflearn.layers.merge_ops import merge
from tflearn.layers.recurrent import lstm
from tflearn.layers.estimator import regression
from pairs_trading import trial, DELAY
from dummy_data import MEAN, DECAY, gen_sim_data, STDEV
from web_data import gen_spread

N = 50000
LEN_HISTORY = 10
PROFIT_THRESHOLD = 3


def create_data_pairs(n, size, src):
	x = []
	y = []
	for i in range(2, n+2):
		seq = src[i:i+size]
		moves = gen_moves(seq)
		x.append(seq)
		y.append(moves)
	return x, y


def gen_moves(seq):
	profit = 0

	while profit < PROFIT_THRESHOLD:
		moves = [0] * len(seq)

		def should_buy(history):
			move = random.randint(0, 1)
			moves[len(history)] = move
			return move == 1

		def should_sell(history):
			move = random.randint(0, 2)
			moves[len(history)] = move
			return move == 2

		profit = trial([should_buy, should_sell, ""], seq)

	return moves


def revenue(preds, values):
	return tf.reduce_mean(values * (preds - tf.fill(tf.shape(preds), 0.5)))


def get_network():
	network = input_data(shape=[None, LEN_HISTORY], name="inp")
	network = reshape(network, [tf.shape(network)[0], LEN_HISTORY, 1], name="added_inp_dim")
	network = lstm(network, LEN_HISTORY, name="hidden", return_seq=True)
	network = [fully_connected(ts, 1, activation='softmax') for ts in network]
	network = merge(network, "concat", axis=1, name="concat_merge")
	network = network / tf.expand_dims(tf.reduce_mean(network, axis=1), 1, name="normed")
	network = regression(network, optimizer="sgd", loss=revenue)
	return network


def train(src=None):
	if src is None:
		src = gen_sim_data(N + LEN_HISTORY + DELAY + 5, MEAN, DECAY, STDEV)

	n = len(src) - LEN_HISTORY - DELAY - 5

	x, y = create_data_pairs(n, LEN_HISTORY, src)

	x = np.array(x)
	y = np.array(y).reshape([n, 1])

	model = tflearn.DNN(get_network())
	model.fit(x, y, batch_size=100, show_metric=True)

	model.save("rand_ann.tflearn")


def gen_predictors():
	model = tflearn.DNN(get_network())
	model.load("rand_ann.tflearn")

	i = 0.6

	def should_buy(history, i=i):
		if len(history) <= LEN_HISTORY:
			return False
		#  print(model.predict([history[-1*LEN_HISTORY:]]))
		return model.predict([history[-1*LEN_HISTORY:]]) > i

	def should_sell(history, i=i):
		if len(history) <= LEN_HISTORY:
			return False
		#  print(model.predict([history[-1*LEN_HISTORY:]]))
		return model.predict([history[-1*LEN_HISTORY:]]) < 1 - i

	return [should_buy, should_sell, "ann {}".format(i)]

if __name__ == '__main__':
	# train()
	print(gen_spread("MSFT", "GOOG", train=True))
	train(gen_spread("MSFT", "GOOG", train=True))
