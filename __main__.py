# coding=utf-8

import numpy as np

import benchmark
# import pairs_trading_lstm
import pairs_trading_deepq
from pairs_trading import *
from web_data import *


from dummy_data import *

# import pairs_trading_ann

print(len(gen_spread("MSFT", "GOOG")))

# spread = [gen_sim_data(DURATION, MEAN, DECAY, STDEV)]
spread = gen_spread("MSFT", "GOOG", train=False)

# for i in np.arange(0.0, 2.0, 0.2):
# 	for j in np.arange(0.0, 2.0, 0.2):
# 		print(i, j, trial([benchmark.gen_predictors(avg=0, threshold_min=i, threshold_max=j)],
# 		      spread, show_fig=True, log_out=False))

trial([benchmark.gen_predictors(avg=0, threshold_min=1, threshold_max=0)]
      + [pairs_trading_deepq.gen_predictors("ann2", version=7)]
      ,
      spread, show_fig=True)

# gen_sim_data(DURATION, MEAN, DECAY, STDEV))
# [pairs_trading_deepq.gen_predictors("ann2", version=19)],
