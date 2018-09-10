# tensorflow
Over the summer, I worked on applying machine learning techniques to financial engineering, specifically in pairs trading.
In this project, I worked on analyzing fluctuations over longer time scales (days, rather than seconds). The issue with a naive approach is the transaction delay: by the time a fluctuation is identified and action taken, it may have reversed itself, yielding a net loss. To address this, I applied machine learning techniques (including neural networks, LSTMs, and deep Q-learning) to predict such fluctuations and make trades accordingly.

## What’s pairs trading?
Imagine we have two stocks, say, Coca-Cola and Pepsi. In general, factors that affect one of these stocks will affect the other in a similar manner. For instance, if sugar becomes more expensive, both their stocks will drop. However, due to market inefficiencies, these fluctuations may not occur simultaneously.
Pairs trading is the process of generating a profit based on these time differences. If Pepsi (for instance) drops first, we buy a certain value of Pepsi stock and short the same value of Coca-Cola stock. When Coca-Cola drops to match, we make a profit from our short. Conversely, if Pepsi stock rises back to its original value (say, if it had just dropped randomly) we will make a profit from our long position on Pepsi. Roughly speaking, we call the difference between the prices of two stocks the “spread” (after taking into account their proportional difference in value).
The key to pairs trading is that we don’t need to know why the spread has moved, merely that it has done so. This makes the problem very amenable to algorithmic trading.

## My work
### Neural Networks
Using data from 2000-2015, I trained a neural network to predict the future behavior of the spread of various pairs of securities (specifically, exchange-traded funds) based on the spread from the preceding 10 days. This neural network was then applied to data from 2016, buying the spread when it was expected to have reached a minimum, and selling when expecting a maximum.
### Deep Q-learning
Deep Q-learning is a form of reinforcement learning, where a neural network is trained to predict the average profit resulting from each action (formally known as the discounted total future reward) and take the action to maximize it. The training and test data used was the same as above. Though often successful, this technique exhibited a great deal of variability, while not generating significantly more profit than the benchmark.
### Benchmark
The benchmark used as a point of comparison sold the spread when it exceeded a given threshold, and bought it when it dropped below another threshold. These thresholds were optimized using data from 2000-2015.
## Results and limitations
As mentioned, the deep Q-learning technique only marginally outcompeted the benchmark algorithm, while exhibiting a great deal of variability. However, the simpler neural network approach significantly outcompeted the benchmark. As shown, when trading the spread of the Australian (EWA) and South African (EZA) Exchange Traded Funds (ETFs), the neural network generated more than twice the profit / share as the benchmark did. I obtained similar results for a variety of co-integrated securities. However, significant losses can be incurred when two securities cease to be co-integrated—I’m presently working on ways to algorithmically identify such cases, in order to protect any previously-generated profits.
