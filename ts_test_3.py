# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:49:29 2018

@author: istravi
"""
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA, AR
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from lightgbm import LGBMRegressor

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model

def _load_data(data):
    """
    data should be pd.DataFrame()
    """
    n_prev = 100
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data['price'].iloc[i+n_prev]) #.as_matrix())
    if not docX:
        pass
    else:
        alsX = np.array(docX)
        alsY = np.array(docY)
        return alsX, alsY

#from statsmodels.tsa.seasonal import seasonal_decompose

filename = "~/Documents/ElementGroup/Data Bitfinex BTCUSD Raw 2018-03-12 10-33-01.000_2018-04-05 07-50-34.449.csv"

df=pd.read_csv(filename, index_col=0, parse_dates=True)

print(df.head())

#bitdata = df.loc[df['coin_id'] == 'bitcoin']

bitdata = df.resample('1T').agg({'amount': np.sum, 'price': np.mean})

pricemax = bitdata.max()['price']
pricemin = bitdata.min()['price']

bitdata=(bitdata-bitdata.min())/(bitdata.max()-bitdata.min())

print(bitdata.head())

#number_of_lags = 10

#for i in range(number_of_lags):
#    bitdata['pricelag' + str(i)] = bitdata.price.shift(i)
#    bitdata['amountlag' + str(i)] = bitdata.amount.shift(i)
#bitdata.dropna(inplace = True)
 
bitdata_training = bitdata.loc['2018-03-23':'2018-03-23']
bitdata_validation = bitdata.loc['2018-03-24':'2018-03-24']


xtr = bitdata_training.drop(columns = ['amount', 'price']).values
xts = bitdata_validation.drop(columns = ['amount', 'price']).values
ytr = bitdata_training['price'].values
yts = bitdata_validation['price'].values

print(bitdata_training.columns)

xs, ys = _load_data(bitdata)

xtr = xs[:2000]
xts = xs[2000:2100]
ytr = ys[:2000]
yts = ys[2000:2100]

#mdl = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
#mdl = LGBMRegressor(n_estimators=1000, learning_rate=0.01)
#mdl.fit(xtr, ytr)    
#forecast = mdl.predict(xts)

epochs = 250

batch_size = len(bitdata_training)
batch_size = 100

model = Sequential()
model.add(LSTM(input_shape=xtr[0].shape, units=250, return_sequences=True))
#model.add(LSTM( input_shape=(20,) , units=100, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(30, return_sequences=False))
model.add(Dropout(0.25))
model.add(Dense(units=1))
model.add(Activation("relu"))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse', 'accuracy'])
#model.compile(loss='val_mean_absolute_error', optimizer='adam', metrics=['mae', 'mse', 'accuracy'])

history = model.fit(
    xtr,
    ytr,
    batch_size=batch_size,
    epochs=epochs,
    callbacks = [EarlyStopping(monitor='mean_squared_error', min_delta=1e-06, patience=5, verbose=1, mode='auto'),],
    verbose=2,
    validation_split=0.05 )

forecast = model.predict(xts)

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')

forecast = pricemin + forecast * (pricemax-pricemin)
yts = pricemin + yts * (pricemax-pricemin)

#bitret = bitdata.price / bitdata.price.shift(1)  - 1
# 
#bitret = bitret.shift(-1)
#  
#print(bitret.head())     
##bitret = bitret.loc['2018-04-01':'2018-04-01']
#
#bitret_training = bitret.loc['2018-03-23':'2018-03-23']
#bitret_validation = bitret.loc['2018-03-24':'2018-03-24']
#
#
#
#
#
#start_pos = int(len(bitret_training)/2)
#
#forecast = []
#
##for i in range(start_pos,len(bitret_training),1):
#for i in range(start_pos,start_pos+50,1):
#    model = ARIMA(bitret_training[0:i], order = (3,0,0))
#    model_fit = model.fit()
#    forecast_point = model_fit.forecast(steps=1)[0][0]
#    forecast.append(forecast_point)
#
##bit_season = seasonal_decompose(bitret)
#                
#fig = plt.figure()
#
#plt.plot(bitret_training)
#
#
#plot_acf(bitret_training, lags=50)
#plot_pacf(bitret_training, lags = 50)
#
#fig.show()


#test_res = adfuller(bitret_training)
#print(test_res) 

#def objfunc(order, endog):
#    fit = ARIMA(endog, order).fit()
#    print("Trying %s, %.4f" % (str(order), fit.aic))
#    return fit.aic
#
#from scipy.optimize import brute
#grid = (slice(1, 10, 1), slice(0, 3, 1), slice(0, 1, 1))
#resbrute = brute(objfunc, grid, args=(bitret,), finish=None)
#
#print(resbrute)


#forecast_length = 10
#model = ARIMA(bitret_training, order = (1,0,0))
#model_fit = model.fit()


#from arch import arch_model
#am = arch_model(bitret_training, vol='Garch', p=1, o=0, q=1, dist='Normal')
#model_fit = am.fit()



#print(model_fit.summary())
              
#forecast = model_fit.forecast(steps=forecast_length)[0]
#print(bitret_validation[0:forecast_length])
#print(forecast)

#forecast = model_fit.forecast(horizon=forecast_length)

fig = plt.figure(figsize=(24,10))

ax = fig.add_subplot(111)
#plt.xlabel('Number of requests every 10 minutes')
linelegends = []
funcnames = []

line, = ax.plot(forecast, color='blue')

linelegends.append(line)
funcnames.append("Forecast")

#line, = ax.plot(bitret_validation[0:forecast_length].values, color='red', label='Real')
#line, = ax.plot(bitret_training[start_pos:start_pos+50].values, color='red', label='Real')
line, = ax.plot(yts, color='red', label='Real')


linelegends.append(line)
funcnames.append("Real")
#h1, l1 = ax1.get_legend_handles_labels()
#h2, l2 = ax2.get_legend_handles_labels()

ax.legend(linelegends, funcnames, loc = 0)

#plt.legend(h1+h2, l1+l2, loc=2)
plt.show()

fig.savefig('test.pdf')

#model = AR(bitret)

#model_fit = model.fit(disp=0)
#model_fit = model.fit()

#print(model_fit.summary())              
              
#fig = bit_season.plot()
#fig.set_size_inches(15, 8)
#fig.show()


#hist = bitret.hist(bins=30)




