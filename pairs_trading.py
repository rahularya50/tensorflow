# coding=utf-8

import matplotlib.pyplot as plt

DELAY = 1


def get_profits(data, should_buy, should_sell, prev_profit=0):
	profits = [prev_profit]
	net_value = [prev_profit]
	will_be_active = False
	active = False
	todo = []
	history = []
	for index, dat in enumerate(data):
		act = 0
		history.append(dat)
		if todo:
			if todo[0][0] == index:
				act = todo.pop(0)[1]
				active += act
		profits.append(profits[-1] - dat * act)
		net_value.append(profits[-1] + active * dat)

		if will_be_active and should_sell(history):
			todo.append((index + DELAY, -1))
			will_be_active = False
		elif not will_be_active and should_buy(history):
			todo.append((index + DELAY, 1))
			will_be_active = True

	if will_be_active:
		profits.append(profits[-1] + data[-1])
		# net_value.append(profits[-1])
	return net_value


def trial(models, spreads, show_fig=True, log_out=True):
	out = []
	if show_fig:
		fig, ax = plt.subplots()
		ax.plot(sum(spreads, []))
		print("\n".join(map(str, sum(spreads, []))))
		ax2 = ax.twinx()
	for model in models:
		profit = 0
		results = []
		for spread in spreads:
			results.extend(get_profits(spread, model[0], model[1], profit))
			profit = results[-1]
		out.append(results[-1])
		print("\n".join(map(str, results)))
		if show_fig:
			ax2.plot(range(len(results)), results, "r-")
		if log_out:
			print(model[2], results[-1])
	if show_fig:
		plt.show()
	return out
