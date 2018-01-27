# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 14:58:15 2018

@author: cosmo
"""

class Portfolio():
    def __init__(self, start_amt, possible_coins, trading_fee=0):
        self.cash = start_amt
        self.held = dict((c, 0) for c in possible_coins)
        self.fee = trading_fee
        self.hist = []
    
    def deposit(self, amt):
        self.cash += amt
        
    def withdraw(self, amt):
        self.cash -= amt
    
    def buy(self, coin, amt, fee=None):
        if fee is None:
            fee = self.fee
        assert self.cash + 0.0001 >= amt*(1+fee)
        self.held[coin] += amt
        self.cash = max(0, self.cash - amt*(1+fee))
    
    def sell(self, coin, amt=None, fee=None):
        if amt is None:
            amt = self.held[coin]
        if fee is None:
            fee = self.fee
        assert self.held[coin] + 0.0001 >= amt
        self.cash += amt*(1-self.fee)
        self.held[coin] = max(0, self.held[coin] - amt)
        
    def update_price(self, coin, change):
        self.held[coin] *= (1+change)
    
    def value(self):
        return self.cash + sum(self.held.values())
    
    def snapshot(self, t = None):
        self.hist.append({'t': t, 'value': self.value(), 'held': dict(self.held)})
