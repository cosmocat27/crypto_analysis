# coding: utf-8
"""
Created on Sun Feb 18 10:20:23 2018

@author: cosmo

Modeling of coin price data based on past performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies import *


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
exclude = ['USDT', 'ADA', 'QTUM', 'TRX', 'ICX', 'LSK'] + not_tradeable

coins = [c for c in coins if c not in exclude]

############### load data ######################

c_hist_in = pd.read_csv('hourly_data_2017_top20.csv', header=[0,1], index_col=0)

prices = pd.concat([c_hist_in[c]['close'] for c in coins], axis=1)
prices.columns = coins


############# prepare datasets #################

# change in price past X hours
price_chg = []
for i in range(1, 8):
    new_df = prices.copy()
    for c in coins:
        new_df[c] = prices[c].shift((i-1)*24) / prices[c].shift(i*24) - 1
    price_chg.append(new_df)

for i in range(1, 7):
    new_df = prices.copy()
    for c in coins:
        new_df[c] = prices[c] / prices[c].shift(i*4) - 1
    price_chg.append(new_df)

# rank of coin based on price change past X hours
price_rnk=[]
for df in price_chg:
    new_df = df.copy()
    for t in df.index:
        a = df.loc[t]
        rnk = a.rank(ascending=False)
        new_df.loc[t] = rnk
    price_rnk.append(new_df)

# target vars: y_chg is % change next 24 hours, y_rnk is rank of y_chg
y_chg = price_chg[0].copy()
y_rnk = price_rnk[0].copy()
for c in coins:
    y_chg.loc[:, c] = price_chg[0].loc[:, c].shift(-24)
    y_rnk.loc[:, c] = price_rnk[0].loc[:, c].shift(-24)

for i, df in enumerate(price_chg):
    df = price_chg[i]
    df = df.drop(df.index[:168], axis=0)
    df = df.drop(df.index[-24:], axis=0)
    df.columns=['chg'+str(i+1)+c for c in coins]
    df['id'] = df.index
    df = pd.wide_to_long(df, stubnames='chg'+str(i+1), i="id", j="coin", suffix='\D+')
    price_chg[i] = df

for i, df in enumerate(price_rnk):
    df = price_rnk[i]
    df = df.drop(df.index[:168], axis=0)
    df = df.drop(df.index[-24:], axis=0)
    df.columns=['rnk'+str(i+1)+c for c in coins]
    df['id'] = df.index
    df = pd.wide_to_long(df, stubnames='rnk'+str(i+1), i="id", j="coin", suffix='\D+')
    price_rnk[i] = df

y_chg = y_chg.drop(y_chg.index[:168], axis=0)
y_rnk = y_rnk.drop(y_rnk.index[:168], axis=0)
y_chg = y_chg.drop(y_chg.index[-24:], axis=0)
y_rnk = y_rnk.drop(y_rnk.index[-24:], axis=0)
y_chg.columns=['chg_ahead_24'+c for c in coins]
y_rnk.columns=['rnk_ahead_24'+c for c in coins]
y_chg['id'] = y_chg.index
y_rnk['id'] = y_rnk.index
y_chg = pd.wide_to_long(y_chg, stubnames='chg_ahead_24', i="id", j="coin", suffix='\D+')
y_rnk = pd.wide_to_long(y_rnk, stubnames='rnk_ahead_24', i="id", j="coin", suffix='\D+')

comb_chg = pd.concat([df for df in price_chg], axis=1)
comb_rnk = pd.concat([df for df in price_rnk], axis=1)
X = pd.concat([comb_chg, comb_rnk], axis=1)

# define y_choice target variable as price change > 0
y_choice = y_rnk.copy()
y_choice['rnk_ahead_24'] = (y_rnk['rnk_ahead_24'] <= 15) & (y_chg['chg_ahead_24'] > 0)
y_choice['rnk_ahead_24'] = y_choice['rnk_ahead_24'].astype(int)
y_choice.columns = ['hold']

# only choose data once a day at UTC 22:00
X.index = list(X.index.levels[0].values)*13
X = X.loc[np.array([int(x.split(":")[0].split(" ")[-1]) for x in X.index]) == 22]
X = X.loc[np.array([int(x.split("/")[0]) for x in X.index]) < 3]

y_choice.index = list(y_choice.index.levels[0].values)*13
y_choice = y_choice.loc[np.array([int(x.split(":")[0].split(" ")[-1]) for x in y_choice.index]) == 22]
y_choice = y_choice.loc[np.array([int(x.split("/")[0]) for x in y_choice.index]) < 3]

y_chg.index = list(y_chg.index.levels[0].values)*13
y_chg = y_chg.loc[np.array([int(x.split(":")[0].split(" ")[-1]) for x in y_chg.index]) == 22]
y_chg = y_chg.loc[np.array([int(x.split("/")[0]) for x in y_chg.index]) < 3]


####################### Model Fitting ########################

from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


X_train, X_test, y_train, y_test, y_val_train, y_val_test = model_selection.train_test_split(X, y_choice, y_chg)

logreg = LogisticRegression()
logreg.fit(X_train, y_train.values.ravel())

y_pred_train_prob = np.array([p[1] for p in logreg.predict_proba(X_train)])
y_pred_test_prob = np.array([p[1] for p in logreg.predict_proba(X_test)])

# distribution of predicted probabilities
a = pd.Series(y_pred_train_prob)
plt.hist(a, bins=20)


# accuracy and performance scores

#y_pred_train = logreg.predict(X_train)
y_pred_train = (y_pred_train_prob > 0.55).astype(int)
accuracy = metrics.accuracy_score(y_pred_train, y_train)
print("logreg training:", accuracy)
metrics.confusion_matrix(y_train, y_pred_train)

#y_pred_test = logreg.predict(X_test)
y_pred_test = (y_pred_test_prob > 0.55).astype(int)
accuracy = metrics.accuracy_score(y_pred_test, y_test)
print("logreg test:", accuracy)
metrics.confusion_matrix(y_test, y_pred_test)

# training -- model
chosen = pd.DataFrame(y_val_train.iloc[np.array(y_pred_train) > 0, 0])
chosen = chosen.groupby(chosen.index)['chg_ahead_24'].mean()
score = 10000
for i in chosen:
    #print(i+1, score)
    score *= i+1
print("model score (training):", score)

# training -- guessing
chosen = pd.DataFrame(y_val_train.iloc[:, 0])
chosen = chosen.groupby(chosen.index)['chg_ahead_24'].mean()
score = 10000
for i in chosen:
    #print(i+1, score)
    score *= i+1
print("guessing score (training):", score)

# test -- model
chosen = pd.DataFrame(y_val_test.iloc[np.array(y_pred_test) > 0, 0])
chosen = chosen.groupby(chosen.index)['chg_ahead_24'].mean()
score = 10000
for i in chosen:
    #print(i+1, score)
    score *= i+1
print("model score (training):", score)

# test -- guessing
chosen = pd.DataFrame(y_val_test.iloc[:, 0])
chosen = chosen.groupby(chosen.index)['chg_ahead_24'].mean()
score = 10000
for i in chosen:
    #print(i+1, score)
    score *= i+1
print("guessing score (testing):", score)

chosen = y_chg.loc[(X.rnk11 <= 4)]
means = chosen.groupby(chosen.index)['chg_ahead_24'].mean()
counts = chosen.groupby(chosen.index)['chg_ahead_24'].count()
chosen = pd.concat([means, counts], axis=1)
chosen.columns = ['avg', 'count']
score = 10000
trades = 0
days = 0
for i in chosen.index:
    #print(i+1, score)
    score *= chosen.loc[i, 'avg']+1
    trades += chosen.loc[i, 'count']
    days += 1
print(score, trades, days)

chosen = y_chg
means = chosen.groupby(chosen.index)['chg_ahead_24'].mean()
counts = chosen.groupby(chosen.index)['chg_ahead_24'].count()
chosen = pd.concat([means, counts], axis=1)
chosen.columns = ['avg', 'count']
score = 10000
trades = 0
days = 0
for i in chosen.index:
    #print(i+1, score)
    score *= chosen.loc[i, 'avg']+1
    trades += chosen.loc[i, 'count']
    days += 1
print(score, trades, days)
