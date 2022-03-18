# Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime as dt
from copy import deepcopy

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Globals
# Hello World Part 2
# REQUEST URL Vars, These are default values. The UI will change them before making the reuquest.
global COIN_SYMBOL; COIN_SYMBOL = 'BTC'
global PERIOD_ID; PERIOD_ID = '1DAY'
global START_DATE; START_DATE = '2015-01-01'
global END_DATE; END_DATE = '2016-02-01'
global LIMIT; LIMIT = 1000
global API_KEY; API_KEY = '53663783-E96C-4CF7-A3F4-0F8D59946927'
global API_KEY2; API_KEY2 = '9117E3A0-8011-4C76-830D-F7BFB6D96199'
global REQUEST_URL; REQUEST_URL = 'https://rest.coinapi.io/v1/exchangerate/{}/USD/history?period_id={}&time_start={}&time_end={}&limit={}&apikey={}&output_format=csv'

# Training
global LOOK_BACK; LOOK_BACK = 10
global NUM_EPOCHS; NUM_EPOCHS = 400
global BATCH_SIZE; BATCH_SIZE = 256
global TT_SPLIT; TT_SPLIT = .67

# convert an array of values into a dataset matrix
def create_dataset(dataset, lookBack=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-lookBack-1):
    a = dataset[i:(i+lookBack), 0]
    dataX.append(a)
    dataY.append(dataset[i + lookBack, 0])
  return np.array(dataX), np.array(dataY)

def main():
  # Import BTC/USD data
  url = REQUEST_URL.format(COIN_SYMBOL, PERIOD_ID, START_DATE, END_DATE, LIMIT, API_KEY)
  print(url)
  data = pd.read_csv('test.csv', sep=';')
  data['date'] = [i[:10] for i in data['time_period_start']]
  print(data)

  # Create Dataframe
  df = deepcopy(data)
  df = df[['rate_open']]
  dataset = df.values
  dataset = dataset.astype('float32')
  print(dataset)

  # Scale Data Frame
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)

  # Define Training and Testing Boundaries
  train_size = int(len(dataset) * TT_SPLIT)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

  # Create Data Sets
  trainX, trainY = create_dataset(train, lookBack=LOOK_BACK)
  testX, testY = create_dataset(test, lookBack=LOOK_BACK)

  try:
    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
  except(Exception):
    print("Not enough data! Choose a larger timeframe or a smaller trading interval.")
    return

  # Build Model
  model = Sequential()
  model.add(LSTM(4, input_shape=(1, LOOK_BACK)))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(trainX, trainY, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2)

  # Make Predictions
  trainPredict = model.predict(trainX)
  testPredict = model.predict(testX)

  # Revert Predictions to Previous Scale
  trainPredict = scaler.inverse_transform(trainPredict)
  trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict)
  testY = scaler.inverse_transform([testY])

  # Print Scores
  trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
  print('Train Score: %.2f RMSE' % (trainScore))
  testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
  print('Test Score: %.2f RMSE' % (testScore))

  # Plot Results
  trainPredictPlot = np.empty_like(dataset)
  trainPredictPlot[:, :] = np.nan
  trainPredictPlot[LOOK_BACK:len(trainPredict) + LOOK_BACK, :] = trainPredict
  testPredictPlot = np.empty_like(dataset)
  testPredictPlot[:, :] = np.nan
  testPredictPlot[len(trainPredict) + (LOOK_BACK * 2) + 1:len(dataset) - 1, :] = testPredict
  plt.plot(df['rate_open'], label='Actual')
  plt.plot(pd.DataFrame(trainPredictPlot, columns=["rate_open"], index=data['date']).rate_open, label='Training')
  plt.plot(pd.DataFrame(testPredictPlot, columns=["rate_open"], index=data['date']).rate_open, label='Testing')
  
  plt.title(label="Coin: {} Interval: {} Epochs: {} Lookback: {} Batchsize: {}".format(COIN_SYMBOL, PERIOD_ID, NUM_EPOCHS, LOOK_BACK, BATCH_SIZE))
  plt.legend(loc='best')
  plt.xticks(np.arange(0, len(data['date']), len(data['date']) / 20 ), rotation=80)
  plt.subplots_adjust(bottom=.265, top=.95, left=.1, right=.98)
  plt.xlabel("Date\nTrain RMSE: %.2f Test RMSE: %.2f" % (trainScore, testScore))
  plt.ylabel("Coin Price")
  plt.show()

if __name__ == "__main__":
    main()