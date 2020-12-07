import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import tensorflow as tf
import keras
import datetime as dt
from datetime import datetime
from getDataFromPoloniex import get_data_from_poloniex
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(7)

data = pandas.read_csv('data/BTC_ETH.csv')[::-1]

data2 = pandas.read_csv('data/BTC_ETH_real.csv')[::-1]
data2.columns = ['date','high','low','open','close','volume','quoteVolume','weightedAverage']

data3 = pandas.read_csv('data/USDT_BTC.csv')[::-1]
data3.columns = ['date','high','low','open','close','volume','quoteVolume','weightedAverage']

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def split_sequence(sequence, n_steps_in, n_steps_out):
	X, Y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		Y.append(seq_y)
	return np.asarray(X), np.asarray(Y)

dataset = tf.data.Dataset(data3.close.ewm(span=14).mean())[::-1]
print("REAL: ")
data4 = pandas.read_csv('data/BTC_USDT_predicted.csv')
datapred = pandas.DataFrame(data4.EMA_14)
print("Predicted: ")
scaler = MinMaxScaler(feature_range=(0, 1))
ds = dataset.to_numpy()
ds1 = datapred.to_numpy()
ds1 = ds1.flatten('F')
ds2 = ds.flatten('F')
ds2 = ds2[0:288]
ds3 = np.abs(ds1-ds2)
print(ds3)
percents = list()
for i in range(288):
    percents.append(100-(ds3[i]*100)/ds2[i])
percents = np.asarray(percents)
print(percents)
dataset = scaler.fit_transform(dataset)
print("DATASET: ")
print(str(type(dataset)))
ds2 = dataset
print(ds2)
train_size = int(len(dataset))
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
print(str(type(train)))
print("Train: ")
trainF = train.flatten('F')
print(str(type(trainF)))
look_back = 3
trainX, trainY = create_dataset(train, look_back)
print("Create Dataset:")
trXXX, trYYY = split_sequence(trr, look_back, 5)
tarrX, tarrY = split_sequence(trainF, look_back, 288)
trXX, trYY = create_dataset(trr, look_back)
print("Split sequence: ")
trXX = np.reshape(trXX, (trXX.shape[0], trXX.shape[1], 1))

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
trXXX = trXXX.reshape(trXXX.shape[0], trXXX.shape[1], 1)
print("Reshape: ")
tarrX = tarrX.reshape(tarrX.shape[0], tarrX.shape[1], 1)
ds2 = ds2.reshape(ds2.shape[0], ds2.shape[1], 1)

trainXX = trainX[3740:len(trainX)]

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)

print(trainXX.shape)
print(tarrX.shape)
trainPredict = model.predict(trainXX)

model1 = Sequential()
model1.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(look_back, 1)))
model1.add(LSTM(100, activation='relu'))
model1.add(Dense(288))
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.fit(tarrX, tarrY, epochs=100, verbose=0)

print(np.asarray([[80], [90], [100]]).shape)
resPredict = model1.predict(tarrX)
resProba = model1.predict_proba(tarrX)
resPred = scaler.inverse_transform(resPredict)
print("PREDICTED: ")
print(resPred)
trainPredict = scaler.inverse_transform(trainPredict)
print("PROBABILITY: ")
print(resProba)

X = list()
for i in range(288):
    X.append(1596525300+300*i)

resPred = resPred.flatten('F')
print(resPred)
trPred = np.flip(resPred, 0)
trPred = trPred[0:288]

df2 = pandas.DataFrame(trPred)
df2['Date'] = [dt.datetime.fromtimestamp(x) for x in X]
df2.reset_index()
df2.columns = ['EMA_14','Date']
print(df2.Date)
resProba = resProba.flatten('F')
resProba = resProba[0:288]
print(len(resProba), len(df2))

plt.plot(df2.Date, resProba)

plt.show()