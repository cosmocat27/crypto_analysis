# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:46:43 2018

@author: cosmo

Simulate trading strategies based on daily price data and compare results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
from datetime import date, datetime
from cryCompare.crycompare import History
from portfolio import Portfolio
from strategies import *


coins = ['BTC', 'ETH', 'XRP', 'BCH', 'ADA', 'LTC', 'XLM', 'IOT', 'TRX',
         'DASH', 'NEO', 'XMR', 'EOS', 'ICX', 'QTUM', 'BTG', 'LSK', 'ETC', 'USDT']
"""
coins = ['BTC', 'ETH', 'XRP', 'BCH', 'ADA', 'LTC', 'XLM', 'IOT', 'TRX', 'DASH',
         'NEO', 'XMR', 'EOS', 'ICX', 'QTUM', 'BTG', 'LSK', 'ETC', 'XRB', 'XEM',
         'USDT', 'VEN', 'PPT', 'OMG', 'STRAT', 'STEEM', 'XVG', 'ZEC', 'BCN', 'BTS',
         'BNB', 'SC', 'SNT', 'ZRX', 'WTC', 'MKR', 'REP', 'KCS', 'WAVES', 'ARDR',
         'DOGE', 'VERI', 'DRGN', 'KMD', 'HSR', 'ETN', 'RHOC', 'AE', 'ARK', 'LRC', 'BAT']
"""

#coins = ['BTC', 'ETH', 'XRP', 'BCH', 'LTC', 'DASH', 'NEO', 'XMR', 'IOT', 
#         'ETC', 'QTUM', 'OMG', 'ADA', 'ZEC', 'LSK', 'XLM', 'STRAT']

not_tradeable = ['XRB', 'XEM', 'STEEM', 'BCN', 'SC', 'MKR', 'REP', 'KCS', 'ARDR',
                 'DOGE', 'VERI', 'DRGN', 'ETN', ' RHOC', 'AE', 'BCC']
exclude = ['QTUM']

coins = [c for c in coins if c not in exclude]

c_hist_in = pd.read_csv('daily_data_2017_top20.csv', header=[0,1], index_col=0)

prices = pd.concat([c_hist_in[c]['close'] for c in coins], axis=1)
prices.columns = coins


start_dt = "2017-11-01"
end_dt = "2018-01-26"
# end_dt = datetime.strftime(today(), "%Y-%m-%d")
start_amt = 10000
trade_fee = 0
#trade_fee = 0.0005

"""
# check price performance over time period    
price_indexed = pd.DataFrame(prices)
for c in prices.columns:
    price_indexed[c] = 100 * prices[c]/(prices[c][0])

plt.figure()
price_indexed[price_indexed.columns[:]].plot(figsize=(12, 9))
plt.legend(loc='best')
print(price_indexed.iloc[-1].sort_values(ascending=False))
"""


# run simulation

# strategy 1: invest start_amt/num_coins in each coin and hodl

hist1 = buy_and_hold(prices, coins, start_amt, start_dt, end_dt)

# strategy 2: distribute evenly across available coins

hist2 = hold_rebalance(prices, coins, start_amt, start_dt, end_dt)

# strategy 3: invest in top 4 by % gain last 24hrs

hist3 = trade_top_n(prices, coins, start_amt, 4, start_dt, end_dt, lag = 1)

hist1 = pd.DataFrame([h['value'] for h in hist1], index=prices.loc[start_dt:end_dt].index, columns=['buy and hold'])
hist2 = pd.DataFrame([h['value'] for h in hist2], index=prices.loc[start_dt:end_dt].index, columns=['hold rebalance'])
hist3 = pd.DataFrame([h['value'] for h in hist3], index=prices.loc[start_dt:end_dt].index, columns=['trade top 4'])
strategy_compare = pd.concat([hist1, hist2, hist3], axis=1)

print(strategy_compare)
plt.figure()
strategy_compare.plot(figsize=(12, 9))
plt.legend(loc='best')






