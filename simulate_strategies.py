# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:46:43 2018

@author: cosmo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
import csv
from datetime import date, datetime
from cryCompare.crycompare import History
from portfolio import Portfolio


coins = ['BTC', 'ETH', 'XRP', 'BCH', 'ADA', 'LTC', 'XLM', 'IOT', 'TRX',
         'DASH', 'NEO', 'XMR', 'EOS', 'ICX', 'QTUM', 'BTG', 'LSK', 'ETC', 'USDT']
"""
coins = ['BTC', 'ETH', 'XRP', 'BCH', 'ADA', 'LTC', 'XLM', 'IOT', 'TRX', 'DASH',
         'NEO', 'XMR', 'EOS', 'ICX', 'QTUM', 'BTG', 'LSK', 'ETC', 'XRB', 'XEM',
         'USDT', 'VEN', 'PPT', 'OMG', 'STRAT', 'STEEM', 'XVG', 'ZEC', 'BCN', 'BTS',
         'BNB', 'SC', 'SNT', 'ZRX', 'WTC', 'MKR', 'REP', 'KCS', 'WAVES', 'ARDR',
         'DOGE', 'VERI', 'DRGN', 'KMD', 'HSR', 'ETN', 'RHOC', 'AE', 'ARK', 'LRC', 'BAT']
"""

coins = ['BTC', 'ETH', 'XRP', 'BCH', 'LTC', 'DASH', 'NEO', 'XMR', 'IOT', 
         'ETC', 'QTUM', 'OMG', 'ADA', 'ZEC', 'LSK', 'XLM', 'STRAT']

not_tradeable = ['XRB', 'XEM', 'STEEM', 'BCN', 'SC', 'MKR', 'REP', 'KCS', 'ARDR',
                 'DOGE', 'VERI', 'DRGN', 'ETN', ' RHOC', 'AE', 'BCC']

c_hist_in = pd.read_csv('daily_data_2017_top20.csv', header=[0,1], index_col=0)

prices = pd.concat([c_hist_in[c]['close'] for c in coins], axis=1)
prices.columns = coins


start_dt = "2017-11-01"
end_dt = "2018-01-26"
# end_dt = datetime.strftime(today(), "%Y-%m-%d")
start_amt = 10000
trade_fee = 0
#trade_fee = 0.0005


pct_chg = pd.DataFrame(index = prices.index, columns = prices.columns)
for c in prices.columns:
    pct_chg[c] = prices[c] / prices[c].shift(1) - 1

prices = prices.loc[start_dt:end_dt]
pct_chg = pct_chg.loc[start_dt:end_dt]

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

p = Portfolio(start_amt, coins, 0)
amt_each = start_amt / len(coins)

for c in coins:
    if prices.iloc[0][c] > 0:
        p.buy(c, amt_each, 0)
p.snapshot(0)

for i in range(1, len(prices.index)):
    for c in coins:
        if prices.iloc[i-1][c] > 0:
            p.update_price(c, pct_chg.iloc[i][c])
        elif prices.iloc[i][c] > 0:
            p.buy(c, amt_each, 0)
    p.snapshot(i)

hist1 = p.hist
#for x in hist1: print(x)


# strategy 2: distribute evenly across available coins

p = Portfolio(start_amt, coins, 0)

amt_each = p.value() / np.sum(prices.iloc[0] > 0)
for c in coins:
    if prices.iloc[0][c] > 0:
        p.buy(c, amt_each, 0)
p.snapshot(0)

for i in range(1, len(prices.index)):
    c_set = [c for c in coins if (prices.iloc[i-1][c] > 0)]
    for c in c_set:
        p.update_price(c, pct_chg.iloc[i][c])
    
    amt_each = p.value() / np.sum(prices.iloc[i] > 0)
        
    for c in c_set:
        if p.held[c] > amt_each:
            p.sell(c, p.held[c] - amt_each)

    for c in c_set:
        if p.held[c] < amt_each:
            p.buy(c, amt_each - p.held[c])
            
    p.snapshot(i)

# confirm via averages
cash = [start_amt]
for i in range(1, len(prices.index)):
    a = pct_chg.iloc[i]
    a = a[a != np.inf]
    a = a[~pd.isnull(a)]
    avg_chg = np.average(a)
    new_amt = cash[-1] * (avg_chg+1)
    cash.append(new_amt)

hist2 = p.hist
#for x in hist2: print(x['t'], x['value'], cash[x['t']])


# strategy 3: invest in top 4 by % gain last 24hrs

p = Portfolio(start_amt, coins, 0)
num_choices = 4

amt_each = p.value() / num_choices
a = pct_chg.iloc[0]
a = a[a != np.inf]
a = a[~pd.isnull(a)]

c_top = a.sort_values(ascending=False)[:num_choices].index
for c in c_top:
    p.buy(c, amt_each, 0)
p.snapshot(0)

for i in range(1, len(prices.index)):
    a = pct_chg.iloc[i]
    a = a[a != np.inf]
    a = a[~pd.isnull(a)]
    
    c_top = a.sort_values(ascending=False)[:num_choices].index
    for c in a.index:
        p.update_price(c, pct_chg.iloc[i][c])
    
    amt_each = p.value() / num_choices
    
    for c in a.index:
        if c not in c_top:
            p.sell(c)
        elif p.held[c] > amt_each:
            p.sell(c, p.held[c] - amt_each)

    for c in c_top:
        if p.held[c] < amt_each:
            p.buy(c, amt_each - p.held[c])
            
    p.snapshot(i)

hist3 = p.hist
#print(hist3)

#for i in range(len(hist1)):
#    print(i, hist1[i]['value'], hist2[i]['value'], hist3[i]['value'])


hist1 = pd.DataFrame([h['value'] for h in hist1], index=prices.index, columns=['buy and hold'])
hist2 = pd.DataFrame([h['value'] for h in hist2], index=prices.index, columns=['hold rebalance'])
hist3 = pd.DataFrame([h['value'] for h in hist3], index=prices.index, columns=['trade top 4'])
strategy_compare = pd.concat([hist1, hist2, hist3], axis=1)

print(strategy_compare)
plt.figure()
strategy_compare.plot(figsize=(12, 9))
plt.legend(loc='best')






