# coding=utf-8
import random
from collections import deque

import keras
import numpy as np
from keras import regularizers
from keras.layers import LSTM
from keras.layers.core import Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

import benchmark
from dummy_data import gen_sim_data
from web_data import gen_spread
from pairs_trading import get_profits, DELAY, trial

N = 500
LEN_HISTORY = 10


class LstmAnn:
	name = "lstm"

	@staticmethod
	def gen_model():
		model = Sequential()
		model.add(LSTM(8, input_shape=(None, 1)))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(2))
		print(model.input_shape)
		print(model.output_shape)
		model.compile(optimizer="adam", loss="mse")
		return model

	@staticmethod
	def pad(inp):
		max_len = max((len(x) for x in inp)) if inp else 0
		for j, x in enumerate(inp):
			inp[j] = np.pad(x, (max_len - len(x), 0), mode="edge")
		return np.reshape(np.array(inp), (-1, max_len, 1))


class Ann:
	name = "ann"

	@staticmethod
	def gen_model():
		model = Sequential()
		model.add(Dense(8, activation='relu', input_shape=(LEN_HISTORY,)))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(2))
		print(model.input_shape)
		print(model.output_shape)
		model.compile(optimizer="adam", loss="mse")
		return model

	@staticmethod
	def pad(inputs):
		return pad_sequences(inputs, maxlen=LEN_HISTORY)


class Ann2:
	name = "ann2"

	@staticmethod
	def gen_model():
		model = Sequential()
		model.add(Dense(8, activation='sigmoid', input_shape=(LEN_HISTORY,), kernel_regularizer=regularizers.l2(0.1)))
		model.add(Dense(8, activation='sigmoid', kernel_regularizer=regularizers.l2(0.1)))
		model.add(Dense(2))
		print(model.input_shape)
		print(model.output_shape)
		model.compile(optimizer="adam", loss="mse")
		return model

	@staticmethod
	def pad(inputs):
		return pad_sequences(inputs, maxlen=LEN_HISTORY)

MODELS = {
	"lstm": LstmAnn,
	"ann": Ann,
	"ann2": Ann2
}


class Memory:
	def __init__(self):
		self.data = deque()

	def store(self, experience):
		self.data.append(experience)
		if len(self.data) > 100:
			self.data.popleft()

	def retrieve(self, n):
		return np.array(self.data)[np.random.choice(len(self.data), size=min(n, len(self.data)), replace=False), :]


def train(train, test, model_class, prev_version=None):
	if prev_version is None:
		models = [model_class.gen_model(), model_class.gen_model()]
		prev_version = 0
	else:
		models = []
		for i in range(2):
			models.append(keras.models.load_model("deepq_{}_{}_{}.h5".format(model_class.name, ["false", "true"][i], prev_version)))

	memory = Memory()

	def pick_move(history, is_active):
		pos = len(history)
		if pos == 0:
			print(pos)
		history = history[-1*LEN_HISTORY:]

		if pos == len(spread):
			action = 1 if is_active else 0
			reward = spread[-1] * action
			experience = ((history, is_active), action, reward, None)
		else:
			if random.random() <= 0.1 or pos < 2:
				action = random.choice([0, 1])
			else:
				q = models[is_active].predict(model_class.pad([history]))
				action = np.argmax(q[0])

			if action == 0:
				reward = 0
			elif is_active:
				reward = spread[min(pos + DELAY, len(spread)) - 1]
			else:
				reward = -1 * spread[min(pos + DELAY, len(spread)) - 1]
			experience = (
				(history, is_active), action, reward, (spread[max(0, pos - LEN_HISTORY + 1):pos + 1], is_active != (action != 0)))
			if not experience[3][0]:
				raise Exception

		memory.store(experience)

		samples = memory.retrieve(5)

		inputs = [[sample[0][0] for sample in samples if not sample[0][1]],
		          [sample[0][0] for sample in samples if sample[0][1]]]
		targets = [[], []]

		for sample in samples:
			new = np.reshape(models[sample[0][1]].predict(model_class.pad([sample[0][0]])), (-1))
			if sample[3] is None:
				new[sample[1]] = sample[2]
			else:
				x = model_class.pad([sample[3][0]])
				new[sample[1]] = (sample[2] + 0.9 * np.amax(models[sample[3][1]].predict(x))) * \
				                 (len(spread) - len(sample[3][0])) / (len(spread) - len(sample[0][0]))
			targets[sample[0][1]].append(new)

		for i, model in enumerate(models):
			if not inputs[i]:
				continue
			model.train_on_batch(model_class.pad(inputs[i]), np.array(targets[i]))

		return action != 0

	def train_should_buy(history):
		return pick_move(history, False)

	def train_should_sell(history):
		return pick_move(history, True)

	def test_should_buy(history):
		q = models[0].predict(model_class.pad([history]))
		return np.argmax(q[0]) != 0

	def test_should_sell(history):
		q = models[1].predict(model_class.pad([history]))
		return np.argmax(q[0]) != 0

	for e in range(50):
		trial([benchmark.gen_predictors(1), [test_should_buy, test_should_sell, str(e)]], train, False, True)
		trial([benchmark.gen_predictors(1), [test_should_buy, test_should_sell, str(e)]], test, False, True)
		for spread in train:
			get_profits(spread, train_should_buy, train_should_sell)
		for i, model in enumerate(models):
			model.save("deepq_{}_{}_{}.h5".format(model_class.name, ["false", "true"][i], e + prev_version + 1))
		print(e)


def gen_predictors(model, version):
	models = []
	for i in range(2):
		models.append(keras.models.load_model("deepq_{}_{}_{}.h5".format(model, ["false", "true"][i], version)))

	def should_buy(history):
		q = models[False].predict(MODELS[model].pad([history]))
		return np.argmax(q[0]) != 0

	def should_sell(history):
		q = models[True].predict(MODELS[model].pad([history]))
		return np.argmax(q[0]) != 0

	return [should_buy, should_sell, "deepq_{}".format(model)]


if __name__ == '__main__':
	# train([gen_sim_data(100, 0, 0.9, 1) for i in range(100)],
	#       [gen_sim_data(100, 0, 0.9, 1) for i in range(10)],
	#       Ann2, prev_version=2)
	train(gen_spread("MSFT", "GOOG", train=True),
	      gen_spread("MSFT", "GOOG", train=False),
	      Ann2)
