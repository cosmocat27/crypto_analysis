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
    # strategy 1: invest an equal amount in each coin and hodl
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


def hold_rebalance(prices_o, coins, start_amt, start_t, end_t, fee=0):
    # strategy 2: distribute cash evenly across available coins
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
    
    return p.hist


def trade_top_n(prices_o, coins, start_amt, num_choices, start_t, end_t, lag = 1, fee=0):
    # strategy 3: distribute investment across the top coins by % gain
    #
    # input:
    # prices : dataframe of price history
    # coins : list of coins (by symbol) available to buy
    # start_amt : initial investment amount
    # num_choices : number of top coins to invest in
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
    
    amt_each = p.value() / num_choices
    a = pct_chg.iloc[0]
    a = a[a != np.inf]
    a = a[~pd.isnull(a)]
    
    c_top = a.sort_values(ascending=False)[:num_choices].index
    for c in c_top:
        p.buy(c, amt_each, 0)
    p.snapshot(0)
    
    for i in range(lag, len(prices.index)):
        a = pct_chg.iloc[i-lag+1]
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
    
    return p.hist
