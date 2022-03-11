# Load Libraries
from typing import final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# Globals
global COIN_SYMBOL; COIN_SYMBOL = 'BTC'
global PERIOD_ID; PERIOD_ID = '1DAY'
global START_DATE; START_DATE = '2021-01-01'
global END_DATE; END_DATE = '2022-02-01'
global API_KEY; API_KEY = '53663783-E96C-4CF7-A3F4-0F8D59946927'
global REQUEST_URL; REQUEST_URL = 'https://rest.coinapi.io/v1/exchangerate/{}/USD/history?period_id={}&time_start={}&time_end={}&apikey={}&output_format=csv'.format(COIN_SYMBOL, PERIOD_ID, START_DATE, END_DATE, API_KEY)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)

def main():
    # Import BTC/USD data
    data = pd.read_csv(REQUEST_URL, sep=';')
    print(data)

    # Create Dataframe
    df = data.loc[(data['rpt_key'] == 'btc_usd')]
    df = df.reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime_id'])
    df = df.loc[df['datetime'] > pd.to_datetime('2017-06-28 00:00:00')]
    df = df[['last']]
    dataset = df.values
    dataset = dataset.astype('float32')

    # Scale Data Frame
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Define Training and Testing Boundaries
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # Create Data Sets
    look_back = 10
    trainX, trainY = create_dataset(train, look_back=look_back)
    testX, testY = create_dataset(test, look_back=look_back)

    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Build Model
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=256, verbose=2)

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
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    plt.plot(df['last'], label='Actual')
    plt.plot(pd.DataFrame(trainPredictPlot, columns=["close"], index=df.index).close, label='Training')
    plt.plot(pd.DataFrame(testPredictPlot, columns=["close"], index=df.index).close, label='Testing')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()