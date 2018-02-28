# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:44:11 2018

@author: cosmo

Functions for implementing various trading strategies for coins
"""

import pandas as pd
import numpy as np
from portfolio import Portfolio


def buy_and_hold(prices_o, coins, start_amt, start_t, end_t, fee=0):
    # invest an equal amount in each coin and hodl
    #
    # input:
    # prices_o : dataframe of price history
    # coins : list of coins (by symbol) available to buy
    # start_amt : initial investment amount
    # start_t, end_t : start and ending indices in price
    # fee : trading fee as a fraction of total transaction (default 0)
    #
    # output:
    # list with total value and amount of each coin held for each time t

    
    pct_chg = pd.DataFrame(index = prices_o.index, columns = prices_o.columns)
    for c in prices_o.columns:
        pct_chg[c] = prices_o[c] / prices_o[c].shift(1) - 1
    
    prices = prices_o.loc[start_t:end_t]
    pct_chg = pct_chg.loc[start_t:end_t]

    p = Portfolio(start_amt, coins, fee)
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
    
    return p.hist


def hold_rebalance(prices_o, coins, start_amt, start_t, end_t, rebal_rate=1, fee=0):
    # distribute cash evenly across available coins
    #
    # input:
    # prices : dataframe of price history
    # coins : list of coins (by symbol) available to buy
    # start_amt : initial investment amount
    # start_t, end_t : start and ending indices in price
    # fee : trading fee as a fraction of total transaction (default 0)
    #
    # output:
    # list with total value and amount of each coin held for each time t
    
    pct_chg = pd.DataFrame(index = prices_o.index, columns = prices_o.columns)
    for c in prices_o.columns:
        pct_chg[c] = prices_o[c] / prices_o[c].shift(1) - 1
    
    prices = prices_o.loc[start_t:end_t]
    pct_chg = pct_chg.loc[start_t:end_t]
    
    p = Portfolio(start_amt, coins, 0)
    
    for i in range(len(prices.index)):
        a = pct_chg.iloc[i]
        a = a[a != np.inf]
        a = a[~pd.isnull(a)]
        
        for c in a.index:
            p.update_price(c, pct_chg.iloc[i][c])
        
        if i % rebal_rate == 0:
            amt_each = p.value() / np.sum(prices.iloc[i] > 0)
            
            for c in a.index:
                if p.held[c] > amt_each:
                    p.sell(c, p.held[c] - amt_each)
        
            for c in a.index:
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
    
    return p.hist


def trade_top_n(prices_o, coins, start_amt, choices, start_t, end_t, lag = 1, hold_for = 1, fee=0):
    # distribute investment across coins by rank in % gain
    #
    # input:
    # prices : dataframe of price history
    # coins : list of coins (by symbol) available to buy
    # start_amt : initial investment amount
    # choices : list of chosen coins by rank
    # start_t, end_t : start and ending indices in price
    # lag : the number of periods to look back to determine % gain (default 1)
    # fee : trading fee as a fraction of total transaction (default 0)
    #
    # output:
    # list with total value and amount of each coin held for each time t
        
    pct_chg = pd.DataFrame(index = prices_o.index, columns = prices_o.columns)
    for c in prices_o.columns:
        pct_chg[c] = prices_o[c] / prices_o[c].shift(1) - 1
    
    prices = prices_o.loc[start_t:end_t]
    pct_chg = pct_chg.loc[start_t:end_t]
    
    p = Portfolio(start_amt, coins, 0)
    if lag > 1:
        for i in range(lag-1):
            p.snapshot(i)
    
    amt_each = p.value() / len(choices)
    a = pct_chg.iloc[0]
    a = a[a != np.inf]
    a = a[~pd.isnull(a)]
    
    c_choose = []
    for c in choices:
        c_choose.append(a.sort_values(ascending=False).index[c])
    
    for c in c_choose:
        p.buy(c, amt_each, 0)
    p.snapshot(lag-1)
    
    for i in range(lag, len(prices.index)):
        a = [1 for x in pct_chg.iloc[i]]
        for j in range(lag):
            a *= 1 + pct_chg.iloc[i-j]
                
        a = a[a != np.inf]
        a = a[~pd.isnull(a)]
        
        c_choose = []
        for c in choices:
            c_choose.append(a.sort_values(ascending=False).index[c])
        
        for c in a.index:
            p.update_price(c, pct_chg.iloc[i][c])
        
        if i % hold_for != 0:
            p.snapshot(i)
            continue
        
        amt_each = p.value() / len(choices)
        
        for c in a.index:
            if c not in c_choose:
                p.sell(c)
            elif p.held[c] > amt_each:
                p.sell(c, p.held[c] - amt_each)
        
        for c in c_choose:
            if p.held[c] < amt_each:
                p.buy(c, amt_each - p.held[c])
        
        p.snapshot(i)
    
    return p.hist


def trade_thresholds(prices_o, coins, start_amt, start_t, end_t, min_buy = -np.inf,
                     max_buy = np.inf, min_sell = -np.inf, max_sell = np.inf, fee = 0):
    # buy/sell if change above/below some threshold % change
    #
    # input:
    # prices : dataframe of price history
    # coins : list of coins (by symbol) available to buy
    # start_amt : initial investment amount
    # choices : list of chosen coins by rank
    # start_t, end_t : start and ending indices in price
    # lag : the number of periods to look back to determine % gain (default 1)
    # fee : trading fee as a fraction of total transaction (default 0)
    #
    # output:
    # list with total value and amount of each coin held for each time t
    
    pct_chg = pd.DataFrame(index = prices_o.index, columns = prices_o.columns)
    for c in prices_o.columns:
        pct_chg[c] = prices_o[c] / prices_o[c].shift(1) - 1
    
    prices = prices_o.loc[start_t:end_t]
    pct_chg = pct_chg.loc[start_t:end_t]
    
    p = Portfolio(start_amt, coins, 0)
    c_held = []
    
    for i in range(len(prices.index)):
        a = pct_chg.iloc[i]
        a = a[a != np.inf]
        a = a[~pd.isnull(a)]
        
        for c in a.index:
            p.update_price(c, pct_chg.iloc[i][c])
        
        c_buy = [c for c in a.index if a[c] <= max_buy and a[c] >= min_buy]
        c_sell = [c for c in a.index if a[c] <= max_sell and a[c] >= min_sell]
        c_held = [c for c in set(c_held+c_buy) if c not in c_sell]
        
        #print(c_buy, c_sell, c_held, a)
        for c in c_sell:
            p.sell(c, p.held[c])
            
        if len(c_held) > 0:
            amt_each = p.value() / len(c_held)
            #print(p.value(), p.cash, amt_each)
            
            for c in c_held:
                if p.held[c] > amt_each:
                    p.sell(c, p.held[c] - amt_each)
            
            for c in c_held:
                if p.held[c] < amt_each:
                    p.buy(c, amt_each - p.held[c])
        
        p.snapshot(i)
    
    return p.hist


def trade_top_n_except(prices_o, coins, start_amt, choices, start_t, end_t, bear_thresh=-0.1, lag = 1, hold_for = 1, fee=0):
    # distribute investment across coins by rank in % gain
    #
    # input:
    # prices : dataframe of price history
    # coins : list of coins (by symbol) available to buy
    # start_amt : initial investment amount
    # choices : list of chosen coins by rank
    # start_t, end_t : start and ending indices in price
    # lag : the number of periods to look back to determine % gain (default 1)
    # fee : trading fee as a fraction of total transaction (default 0)
    #
    # output:
    # list with total value and amount of each coin held for each time t
        
    pct_chg = pd.DataFrame(index = prices_o.index, columns = prices_o.columns)
    for c in prices_o.columns:
        pct_chg[c] = prices_o[c] / prices_o[c].shift(1) - 1
    
    prices = prices_o.loc[start_t:end_t]
    pct_chg = pct_chg.loc[start_t:end_t]
    
    p = Portfolio(start_amt, coins, 0)
    if lag > 1:
        for i in range(lag-1):
            p.snapshot(i)
    
    amt_each = p.value() / len(choices)
    a = pct_chg.iloc[0]
    a = a[a != np.inf]
    a = a[~pd.isnull(a)]
    
    cnt = 0
    avg = np.average(a)
    #print(avg)
    
    c_choose = []
    if avg < bear_thresh:
        cnt += 1
        for c in choices:
            c_choose.append(a.sort_values(ascending=True).index[c])
        #print(a.sort_values(ascending=True))
    else:
        for c in choices:
            c_choose.append(a.sort_values(ascending=False).index[c])
        #print(a.sort_values(ascending=False))
    
    for c in c_choose:
        p.buy(c, amt_each, 0)
    p.snapshot(lag-1)
    
    for i in range(lag, len(prices.index)):
        a = [1 for x in pct_chg.iloc[i]]
        for j in range(lag):
            a *= 1 + pct_chg.iloc[i-j]
                
        a = a[a != np.inf]
        a = a[~pd.isnull(a)]
        
        avg = np.average(a)
        #print(avg)
        
        c_choose = []
        if avg < 1+bear_thresh:
            cnt += 1
            for c in choices:
                c_choose.append(a.sort_values(ascending=True).index[c])
            #print(a.sort_values(ascending=True))
        else:
            for c in choices:
                c_choose.append(a.sort_values(ascending=False).index[c])
            #print(a.sort_values(ascending=False))
        
        for c in a.index:
            p.update_price(c, pct_chg.iloc[i][c])
        
        if i % hold_for != 0:
            p.snapshot(i)
            continue
        
        amt_each = p.value() / len(choices)
        
        for c in a.index:
            if c not in c_choose:
                p.sell(c)
            elif p.held[c] > amt_each:
                p.sell(c, p.held[c] - amt_each)
        
        for c in c_choose:
            if p.held[c] < amt_each:
                p.buy(c, amt_each - p.held[c])
        
        p.snapshot(i)
    
    #print(cnt)
    
    return p.hist
    
    
    
    
    
    
    
    