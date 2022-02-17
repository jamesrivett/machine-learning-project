import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

raw_data = pd.read_csv('BTC-USD-2021.csv')
raw_data['DateTime'] = pd.to_datetime('Date')
print(raw_data)