# coding=utf-8
import keras
import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

from dummy_data import MEAN, DECAY, gen_sim_data, STDEV
from pairs_trading import DELAY
from web_data import gen_spread

N = 50000
LEN_HISTORY = 10


def create_data_pairs(size, src):
	x = []
	y = []
	for i in range(2, len(src)-size-DELAY):
		x.append(src[i:i+size])
		y.append(to_categorical(int(src[i + size + DELAY] > src[i + size + DELAY - 1]), num_classes = 2))
	return x, y


def get_network():
	model = Sequential()
	model.add(Dense(8, activation='sigmoid', input_shape=(LEN_HISTORY,)))
	model.add(Dense(2, activation="softmax"))
	print(model.input_shape)
	print(model.output_shape)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
	return model


def train(src=None):
	if src is None:
		src = [gen_sim_data(N + LEN_HISTORY + DELAY + 5, MEAN, DECAY, STDEV)]

	x, y = [], []
	for batch in src:
		batch_x, batch_y = create_data_pairs(LEN_HISTORY, batch)
		x += batch_x
		y += batch_y

	x = np.array(x)
	y = np.array(y).reshape([-1, 2])

	print(len(x))

	model = get_network()
	model.fit(x, y, batch_size=1000, epochs=10000)

	model.save("ann.h5")


def gen_predictors(i=0.8):
	model = keras.models.load_model("ann.h5")

	def should_buy(history, i=i):
		if len(history) <= LEN_HISTORY:
			return False
		prediction = model.predict(np.array([history[-1*LEN_HISTORY:]]))[0]
		# print(history[-1], prediction)
		return np.argmax(prediction) == 1 and max(prediction) > i

	def should_sell(history, i=i):
		if len(history) <= LEN_HISTORY:
			return False
		prediction = model.predict(np.array([history[-1*LEN_HISTORY:]]))[0]
		# print(history[-1], prediction)
		return np.argmax(prediction) == 0 and max(prediction) > i

	return [should_buy, should_sell, "ann {}".format(i)]

if __name__ == '__main__':
	# train([gen_sim_data(10000, MEAN, DECAY, STDEV)])
	# print(gen_spread("MSFT", "GOOG", train=True))
	train(gen_spread("MSFT", "GOOG", train=True))
