# coding=utf-8
import tensorflow as tf
import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from pairs_trading import DELAY, get_profits
from dummy_data import MEAN, DECAY, gen_sim_data, STDEV
from web_data import gen_spread

N = 500
LEN_HISTORY = 50


def create_data_pairs(n, size, src):
	x = []
	y = []
	for i in range(0, n-size-1):
		seq = src[i:i+size]
		moves = gen_moves(seq)
		x.append(seq)
		y.append([[1 if index == move else 0 for index in range(3)] for move in moves])
	print(y)
	return x, y


def gen_moves(seq):
	best_profit = 0
	best_moves = [0] * len(seq)

	for _ in range(100):
		moves = [0] * len(seq)

		def should_buy(history):
			move = random.choice([0, 1])
			moves[len(history) - 1] = move
			return move == 1

		def should_sell(history):
			move = random.choice([0, 2])
			moves[len(history) - 1] = move
			return move == 2

		profit = get_profits(seq, should_buy, should_sell)[-1]
		if profit > best_profit:
			best_profit = profit
			best_moves = moves

	print(best_profit)

	return best_moves


def gen_model():
	model = Sequential()
	model.add(LSTM(10, return_sequences=True, input_shape=(None, 1)))
	model.add(TimeDistributed(Dense(3, activation='softmax')))
	print(model.input_shape)
	print(model.output_shape)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
	return model


def train(src=None):
	if src is None:
		src = gen_sim_data(N*(LEN_HISTORY + DELAY + 1), MEAN, DECAY, STDEV)

	n = len(src)

	x, y = create_data_pairs(n, LEN_HISTORY, src)

	print(len(x), len(x[0]))

	x = np.array(x)
	y = np.array(y)

	print(x.shape)
	print(y.shape)
	x = x.reshape((x.shape[0], LEN_HISTORY, 1))

	print("Data ready")

	model = gen_model()

	print("Beginning training!")

	print(x.shape)

	model.fit(x, y, batch_size=100, epochs=50)

	model.save("lstm.h5")


def gen_predictors():
	model = keras.models.load_model("lstm.h5")

	def should_buy(history):
		print(model.predict(np.array(history).reshape((1, len(history), 1)))[0][-1])
		return np.argmax(model.predict(np.array(history).reshape((1, len(history), 1)))[0][-1]) == 1

	def should_sell(history):
		print(model.predict(np.array(history).reshape((1, len(history), 1)))[0][-1])
		return np.argmax(model.predict(np.array(history).reshape((1, len(history), 1)))[0][-1]) == 2

	return [should_buy, should_sell, "lstm"]


if __name__ == '__main__':
	train()
	#  train(gen_spread("MSFT", "GOOG", train=True))
