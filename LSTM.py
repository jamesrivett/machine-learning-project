# Load Libraries
from appJar import gui
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
# REQUEST URL Vars, These are default values. The UI will change them before making the reuquest.
global COIN_SYMBOL; COIN_SYMBOL = 'BTC'
global PERIOD_ID; PERIOD_ID = '1DAY'
global START_DATE; START_DATE = '2021-01-01'
global END_DATE; END_DATE = '2022-02-01'
global LIMIT; LIMIT = 1000
global API_KEY; API_KEY = '53663783-E96C-4CF7-A3F4-0F8D59946927'
global REQUEST_URL; REQUEST_URL = 'https://rest.coinapi.io/v1/exchangerate/{}/USD/history?period_id={}&time_start={}&time_end={}&limit={}&apikey={}&output_format=csv'

# Training
global LOOK_BACK; LOOK_BACK = 10
global NUM_EPOCHS; NUM_EPOCHS = 200

app = gui("ML Final Project", " 800x900")

# convert an array of values into a dataset matrix
def create_dataset(dataset, lookBack=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-lookBack-1):
    a = dataset[i:(i+lookBack), 0]
    dataX.append(a)
    dataY.append(dataset[i + lookBack, 0])
  return np.array(dataX), np.array(dataY)

def getPeriod(period):
  values = {"1 Hour": "1HRS", "12 hour": "12HRS", "Day": "1DAY", "Week": "7DAY"}
  return values[period]

def getSymbol(symbol):
  values = {"Bitcoin": "BTC",  "Ethereum": "ETH", "Tether": "USDT", "ADA": "ADA",  "DogeCoin": "DOGE", "AVAX":  "AVAX",  "XTZ": "XTZ",  "ShibaCoin": "SHIB",  "DOT": "DOT", "SOL": "SOL"}

def doPrediction(btn):
  # Harvest Inputs
  StartingDay = app.getEntry("Starting Day:    ")
  StartingMonth = app.getEntry("Starting Month: ")
  StartingYear = app.getEntry("Starting Year:   ")
  EndingDay = app.getEntry("Ending Day:    ")
  EndingMonth = app.getEntry("Ending Month: ")
  EndingYear = app.getEntry("Ending Year:   ")
  period = app.getOptionBox("Trading Interval")
  symbol = app.getOptionBox("Select a Crypto")

  # Build URL Variables
  CON_SYMBOL = getSymbol(symbol)
  PERIOD_ID = getPeriod(period)
  START_DATE = dt.datetime(int(StartingYear), int(StartingMonth), int(StartingDay))
  END_DATE = dt.datetime(int(EndingYear), int(EndingMonth), int(EndingDay))

  

  # Import BTC/USD data
  url = REQUEST_URL.format(COIN_SYMBOL, PERIOD_ID, str(START_DATE.date()), str(END_DATE.date()), LIMIT, API_KEY)
  print(url)
  data = pd.read_csv(url, sep=';')
  print(data)

  # Create Dataframe
  df = deepcopy(data)
  df['datetime'] = pd.to_datetime(df['time_period_start'])
  df = df[['rate_open']]
  dataset = df.values
  dataset = dataset.astype('float32')
  print(dataset)

  # Scale Data Frame
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)

  # Define Training and Testing Boundaries
  train_size = int(len(dataset) * 0.67)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

  # Create Data Sets
  try:
    trainX, trainY = create_dataset(train, lookBack=LOOK_BACK)
    testX, testY = create_dataset(test, lookBack=LOOK_BACK)
  except(Exception):
    print("Not enough data! Choose a larger timeframe or a smaller trading interval.")

  # Reshape input to be [samples, time steps, features]
  trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

  # Build Model
  model = Sequential()
  model.add(LSTM(4, input_shape=(1, LOOK_BACK)))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(trainX, trainY, epochs=NUM_EPOCHS, batch_size=256, verbose=2)

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
  plt.plot(pd.DataFrame(trainPredictPlot, columns=["rate_close"], index=df.index).rate_close, label='Training')
  plt.plot(pd.DataFrame(testPredictPlot, columns=["rate_close"], index=df.index).rate_close, label='Testing')
  plt.legend(loc='best')
  plt.show()


def main():
    # Build UI
    app.setFont(20)
    app.addLabel("l0", "ML interface")
    app.setLabelBg("l0", "blue")
    app.addLabelOptionBox("Select a Crypto",["   - Cryptocurrencies -", "Bitcoin", "Ethereum",
                            "Tether", "ADA", "DogeCoin", "AVAX", "XTZ",
                            "ShibaCoin", "DOT","SOL"])
    app.startLabelFrame("Timeframe for prediction")
    app.setFont(20)
    app.addLabelEntry("Starting Day:    ")
    app.addLabelEntry("Starting Month: ")
    app.addLabelEntry("Starting Year:   ")
    app.setFont(14)
    app.addLabel("a", " ")
    app.addLabelEntry("Ending Day:    ")
    app.addLabelEntry("Ending Month: ")
    app.addLabelEntry("Ending Year:   ")
    app.addLabelOptionBox("Trading Interval", ["1 Hour", "12 hour", "Day", "Week"])
    app.addButton("Predict", doPrediction)
    app.setFont(20)

    app.go()

if __name__ == "__main__":
    main()