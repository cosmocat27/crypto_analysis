# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 11:41:23 2018

@author: cosmo

For selected coins and given hourly historical price data, use pattern matching
on past historical movements to project likely outcome (selects the top X past
price patterns most similar to current performance, based on time series
distance, and provides a projection based on the average selected).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies import *
from datetime import datetime

from ts_functions import *


coins = ['BTC', 'ETH', 'XRP', 'BCH', 'ADA', 'LTC', 'XLM', 'IOT', 'TRX', 'DASH',
         'NEO', 'XMR', 'EOS', 'ICX', 'QTUM', 'BTG', 'LSK', 'ETC', 'XRB', 'XEM',
         'USDT', 'VEN', 'PPT', 'OMG', 'STRAT', 'STEEM', 'XVG', 'ZEC', 'BCN', 'BTS',
         'BNB', 'SC', 'SNT', 'ZRX', 'WTC', 'MKR', 'REP', 'KCS', 'WAVES', 'ARDR',
         'DOGE', 'VERI', 'DRGN', 'KMD', 'HSR', 'ETN', 'RHOC', 'AE', 'ARK', 'LRC', 'BAT']

#coins = ['BTC', 'ETH', 'XRP', 'BCH', 'LTC', 'DASH', 'NEO', 'XMR', 'IOT', 
#         'ETC', 'QTUM', 'OMG', 'ADA', 'ZEC', 'LSK', 'XLM', 'STRAT']

not_tradeable = ['XRB', 'XEM', 'STEEM', 'BCN', 'SC', 'MKR', 'REP', 'KCS', 'ARDR',
                 'DOGE', 'VERI', 'DRGN', 'ETN', ' RHOC', 'AE', 'BCC']

coins = coins[:21]
exclude = ['QTUM', 'USDT', 'ADA', 'TRX', 'ICX', 'LSK'] + not_tradeable

coins = [c for c in coins if c not in exclude]

c_hist_in = pd.read_csv('hourly_data_2017_top20.csv', header=[0,1], index_col=0)

prices = pd.concat([c_hist_in[c]['close'] for c in coins], axis=1)
prices.columns = coins

volumes = pd.concat([c_hist_in[c]['volumeto'] for c in coins], axis=1)
volumes.columns = coins


prices_norm = prices.copy()
for c in prices_norm.columns:
    prices_norm[c] = remove_outliers(prices_norm[c].values)
    #prices_norm[c] = center(prices_norm[c].values, 0)
    prices_norm[c] = scale(prices_norm[c].values, 100)

volumes_norm = volumes.copy()
for c in volumes_norm.columns:
    volumes_norm[c] = remove_outliers(volumes_norm[c].values)
    volumes_norm[c] = center(volumes_norm[c].values, 0)
    volumes_norm[c] = scale(volumes_norm[c].values, 100)
    volumes_norm[c] = smooth_ctr(volumes_norm[c].values, 5)


def get_distances(price_hist, target_pattern, projection_period, step_size = 24):
    window = len(target_pattern.index)
    target_pattern = center(target_pattern, 0)
    distances = []
    
    min_dist = np.inf
    
    for i in range(len(price_hist.index) - projection_period - 1, window-2, -step_size):
        start = price_hist.index[i-window+1]
        end = price_hist.index[i]
        start_proj = price_hist.index[i+1]
        end_proj = price_hist.index[i+projection_period]
        
        candidate_pattern = price_hist[start:end]
        projection = price_hist[start_proj:end_proj]
        candidate_pattern_ctr = pd.Series(center(candidate_pattern, 0), index=candidate_pattern.index)
        dist = eu_distance(target_pattern, candidate_pattern_ctr)
        
        distances.append([(start, end), candidate_pattern.values, projection.values, dist])
        
        if dist < min_dist:
            min_dist = dist
            closest_match = candidate_pattern
            closest_projection = price_hist[start_proj:end_proj]
        #print(start, end, dist, min_dist)
    
    return distances, closest_match, closest_projection


def predict_nearest(price_hist, target_pattern, projection_period, num_neighbors=5):
    window = len(target_pattern.index)
    distances, closest_match, closest_projection = get_distances(price_hist, target_pattern, projection_period)
    distances.sort(key=lambda x: x[3])
    
    graph = pd.Series(target_pattern.values)
    prediction = np.array([0]*(window + projection_period))
    avg_dist = 0
    num_neighbors = min(num_neighbors, len(distances))
    for i in range(num_neighbors):
        pattern = distances[i][1]
        projection = distances[i][2]
        full = np.concatenate([pattern, projection])
        prediction = np.add(prediction, full)
        avg_dist += distances[i][3]
        graph = pd.concat([graph, pd.Series(full)], axis=1)
    
    prediction /= num_neighbors
    avg_dist /= num_neighbors
    
    graph.columns = ['target'] + [str(x) for x in range(1, num_neighbors+1)]
    
    for i in range(window, window + projection_period):
        graph.loc[i, 'target'] = graph.loc[i-1, 'target'] * prediction[i] / prediction[i-1]
    
    #graph = pd.concat([graph, pd.Series(prediction)], axis=1)
    
    #plt.figure()
    #graph.plot(figsize=(12, 9))
    #plt.legend(loc='best')
    
    pred_inc = ((graph.loc[window + projection_period - 1, 'target'] / graph.loc[window-1, 'target']) - 1) * 100
    
    return pred_inc, avg_dist, graph


window = 24
projection_period = 24
offset = 0
num_neighbors = 10
coin = 'XRP'

"""
coins_to_predict = ['BTC', 'ETH', 'LTC']
for c in coins_to_predict:
    target_pattern = prices_norm.loc[prices_norm.index[-window:], c]
    pred_inc, avg_dist, graph = predict_nearest(prices_norm.loc[:, c], target_pattern, projection_period, 10)
    print("predicted increase ({}): {}%, avg_dist: {}".format(c, pred_inc, avg_dist))
"""

#def apply_sim(prices, prices_norm, coin, window, projection_period):

predictions = []
m = len(prices.index)
for i in range(window+projection_period+offset, m-projection_period, 24):
    target_pattern = prices_norm.loc[prices_norm.index[i-window:i], coin]
    price_hist = prices_norm.loc[prices.index[:i], coin]
    pred, avg_dist, graph = predict_nearest(price_hist, target_pattern, projection_period, num_neighbors)
    actual = 100 * (prices_norm.loc[prices.index[i+projection_period-1], c] / prices_norm.loc[prices.index[i-1], c] - 1)
    
    if avg_dist > 150:
        plt.figure()
        graph.plot(figsize=(12, 9))
        plt.legend(loc='best')
    
    predictions.append([prices.index[i-1], pred, actual, avg_dist])
    #print(prices.index[i-1], pred, actual, avg_dist)


predictions = pd.DataFrame(predictions, columns = ['time', 'pred', 'actual', 'avg_dist'])

dist_thresh = np.percentile(predictions.avg_dist, 75)
plt.scatter(predictions.pred[predictions.avg_dist < dist_thresh], predictions.actual[predictions.avg_dist < dist_thresh])
predictions[predictions.avg_dist < dist_thresh].corr()


