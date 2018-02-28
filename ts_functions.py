# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 10:20:23 2018

@author: cosmo

Various helper functions for manipulating time series data
"""

from math import sqrt
import numpy as np


def eu_distance(ts1, ts2):
    total = 0
    for i in range(len(ts1)):
        total += (ts1[i] - ts2[i]) ** 2
    return sqrt(total)


def remove_outliers(ts):
    q1 = np.percentile(ts, 25)
    q3 = np.percentile(ts, 75)
    iqr = q3 - q1
    hi = q3 + 1.5*iqr
    lo = q1 - 1.5*iqr
    
    ts = [min(hi, max(lo, x)) for x in ts]
    return ts


def normalize(ts):
    m = np.mean(ts)
    std = np.std(ts)
    
    if std > 0:
        ts = [(x-m)/std for x in ts]
    else:
        ts = [x-m for x in ts]
    return ts


def center(ts, ctr):
    m = np.mean(ts)
    ts = [x-(m-ctr) for x in ts]
    return ts


def scale(ts, rng):
    hi = max(ts)
    lo = min(ts)
    ts = [x*rng/(hi-lo) for x in ts]
    return ts


def smooth_ctr(ts, period):
    smoothed = ts[:]
    for i, x in enumerate(ts):
        j = 1
        total = ts[i]
        n = 1
        while j <= period and i - j >= 0 and i + j < len(ts):
            total += ts[i-j] + ts[i+j]
            n += 2
            j += 1
        smoothed[i] = total / n
        
    return smoothed
        

def ma_lag(ts, period):
    ma = [None] * len(ts)
    for i, x in enumerate(ts):
        if i < period:
            continue
        total = 0
        for j in range(0, period+1):
            total += ts[i-j]
        ma[i] = total / (period+1)
    
    return ma

