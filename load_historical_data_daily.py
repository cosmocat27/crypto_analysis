# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 17:16:43 2018

@author: cosmo
"""

import pandas as pd
import numpy as np
from sklearn import model_selection
from datetime import date, datetime
from cryCompare.crycompare import History

#p = Price()
# price_curr = p.priceMulti(coins, 'USD')

coins = ['BTC', 'ETH', 'XRP', 'BCH', 'ADA', 'LTC', 'XLM', 'IOT', 'TRX',
         'DASH', 'NEO', 'XMR', 'EOS', 'ICX', 'QTUM', 'BTG', 'LSK', 'ETC']

h = History()
d_now = date.today()
d0 = date(2017, 1, 1)
num_days = (d_now-d0).days + 1

df_dict = {}
for c in coins:
    histo = h.histoDay(c, 'USD', limit=num_days)
    if histo['Data']:
        df_histo = pd.DataFrame(histo['Data'])
        df_histo['time'] = pd.to_datetime(df_histo['time'], unit='s')
        df_histo.index = df_histo['time']
        del df_histo['time']
        
        df_dict[c] = df_histo

c_hist = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())

c_hist.to_csv(path_or_buf = 'daily_data_2017_top20.csv', index=True)
