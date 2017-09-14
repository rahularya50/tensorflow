# coding=utf-8

import benchmark
# import pairs_trading_lstm
import pairs_trading_ann
from pairs_trading import *
from process_data import make_spreads

# import pairs_trading_ann

# spread = [gen_sim_data(DURATION, MEAN, DECAY, STDEV)]
spread = make_spreads(train=False)[6]
print(len(spread), len(spread[0]))

# for i in np.arange(0.0, 2.0, 0.2):
# 	for j in np.arange(0.0, 2.0, 0.2):
# 		print(i, j, trial([benchmark.gen_predictors(avg=0, threshold_min=i, threshold_max=j)],
# 		      spread, show_fig=True, log_out=False))

trial([benchmark.gen_predictors()]
      + [pairs_trading_ann.gen_predictors()]
      ,
      spread, show_fig=True)

# gen_sim_data(DURATION, MEAN, DECAY, STDEV))
# [pairs_trading_deepq.gen_predictors("ann2", version=19)],
