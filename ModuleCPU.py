import tensorflow as tf
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
import pandas as pd
import time
from getDataFromPoloniex import get_data_from_poloniex

tf.random.set_seed(13)

df = pd.read_csv('data/USDT_BTC.csv')
df.columns = ['date','high','low','open','close','volume','quoteVolume','weightedAverage']
df['ema'] = df.close.ewm(span=14).mean()

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []
  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size
  #print(history_size)
  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])
    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])
  return np.array(data), np.array(labels)
 
features_considered = ['open', 'close', 'ema']
features = df[features_considered]
features.index = df['date']

TRAIN_SPLIT = 300000
BATCH_SIZE = 256
BUFFER_SIZE = 10000
STEP = 1
EVALUATION_INTERVAL = 200
EPOCHS = 10
future_target = 36
past_history = 720

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std


x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

print ('Single window of past history : {}'.format(x_train_multi[0].shape))
print ('\n Target EMA to predict : {}'.format(y_train_multi[0].shape))

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(36))
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

start_time = time.time()
multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)

total_time = time.time() - start_time
multi_step_model.save('data/trained_model')
print('Total time of fitting on CPU: ', total_time)

for x, y in val_data_multi.take(3):
    pred = multi_step_model.predict(x)[0]
    
pred = pred*data_std[0]+data_mean[0]

print("PREDICTIONS FOR PLOTTING: ")
print(pred)
