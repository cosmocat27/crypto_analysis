# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 20:24:09 2018

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
num_hours = (d_now-d0).total_seconds()/3600

df_dict = {}
for c in coins:
    histo = h.histoHour(c, 'USD', limit=num_hours)
    if histo['Data']:
        df_histo = pd.DataFrame(histo['Data'])
        df_histo['time'] = pd.to_datetime(df_histo['time'], unit='s')
        df_histo.index = df_histo['time']
        del df_histo['time']
        del df_histo['volumefrom']
        del df_histo['high']
        del df_histo['low']
        del df_histo['open']
        df_histo['volumeto']
        
        df_dict[c] = df_histo

c_hist = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())

c_hist.to_csv(path_or_buf = 'hourly_data_2017_top20.csv', index=True)
