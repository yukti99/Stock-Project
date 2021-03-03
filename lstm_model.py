# Importing Important libraries

# For data cleaning and visualization
import pandas as pd
import numpy as np
from numpy import array
from datetime import date
from datetime import datetime, timedelta
import pandas_datareader.data as web

# For creating model
import tensorflow as tf
import keras
from keras import optimizers, callbacks
from keras.models import Sequential, Model
from keras.layers import InputLayer, Input, Masking
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

# For saving the model
import pickle

# For model Evaluation
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
import math
# For Plotting
import matplotlib.pyplot as plt
from pylab import rcParams
import plotly.express as px
import plotly.graph_objects as go
import plotly
import plotly.offline as offline
pd.options.plotting.backend = "plotly"

# Helper Functions


def getTickerData(ticker, start, end):
  # exception handling for unexpected errors
  try:
    df = web.DataReader(ticker, 'yahoo', start, end)
    # convert the web data to csv format
    df.to_csv('dataframes/{}.csv'.format(ticker), index=True)
  except:
    print("Error in getting Data...")


def get_data(start_date, end_date, company, stkMarket=""):
  # Ticker Information
  # Ticker Examples = ['MMM',  'ABT',  'AAPL',  'AMAT',  'APTV',  'ADM',  'ARNC',  'AMZN', 'GOOG']
  # Stock data retrieval using yahoo
  ticker = company
  stockMarket = stkMarket
  if stockMarket:
    ticker = ticker + "." + stockMarket
    print("Ticker = ", ticker)
  print(f"Collecting the required data for {ticker} from dates: {start_date} to {end_date}...")
  getTickerData(ticker, start_date, end_date)
  df = web.DataReader(ticker, 'yahoo', start_date, end_date)
  df = pd.read_csv('dataframes/{}.csv'.format(ticker))
  return df


def data_cleaning(df):
  df1 = df.drop(columns=["High", "Low", "Open", "Volume", "Adj Close"])
  df1['Date'] = pd.to_datetime(df['Date'])
  # Setting index as date
  df1.index = df1['Date']
  # Creating dataframe
  data = df1.sort_index(ascending=True, axis=0)
  data = data.drop(columns=['Date'])
  return data


def create_dataset(dataset, time_step=1):
  dataX, dataY = [], []
  for i in range(len(dataset) - time_step):
    a = dataset[i:(i+time_step), 0]  # i = 0,1,2,3 .... i-99   100
    dataX.append(a)
    dataY.append(dataset[i+time_step, 0])
  return np.array(dataX), np.array(dataY)


def split_train_test(d, time_step=1, split_size=0.65):
  # splitting the dataset into train and test split
  # let us split 65% training data and 35% test data
  train_size = int(len(d)*split_size)
  test_size = len(d) - train_size
  train_data, test_data = d[0:train_size, :], d[train_size:len(d), :1]

  # reshape into X=t, t+1, t+2, t+3 and Y = t+4
  X_train, y_train = create_dataset(train_data, time_step)
  X_test, y_test = create_dataset(test_data, time_step)
  return (train_data, test_data, X_train, y_train, X_test, y_test)


def reshape_data(X_train, X_test):
  # reshape input to be [samples, time_steps, features] which is required for LSTM
  X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
  X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
  return (X_train, X_test)


def LSTM_Model(X_train, y_train, features=1):

  # Initializing the RNN
  model = Sequential()

  # Add the LSTM layers and some dropout regularization
  model.add(LSTM(units=50, return_sequences=True,
                 input_shape=(X_train.shape[1], features)))
  model.add(Dropout(0.2))

  model.add(LSTM(units=50, return_sequences=True))
  model.add(Dropout(0.2))

  model.add(LSTM(units=50, return_sequences=True))
  model.add(Dropout(0.2))

  model.add(LSTM(units=50))
  model.add(Dropout(0.2))

  # Adding the output layer (1-unit output layer)
  model.add(Dense(units=1))

  # Compiling the lstm
  model.compile(optimizer='adam', loss='mean_squared_error')

  # Fit the model to the training set
  hist = model.fit(X_train, y_train, epochs=24, batch_size=32, verbose=2)

  return (hist, model)


def model_loss(model, X, Y):
  return model.evaluate(X, Y, verbose=2)


def plot_training_loss(model_hist):
  # plotting the lose curve during model training
  plt.plot(model_hist.history['loss'])
  plt.title('Training Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train'], loc='upper left')
  plt.show()


def Model_Evaluation(actual_prices, predicted_prices):
  # Mean Absolute Error
  MAE = metrics.mean_absolute_error(actual_prices, predicted_prices)
  # Mean Squared Error
  MSE = metrics.mean_squared_error(actual_prices, predicted_prices)
  # Root Mean Squared Error
  RMSE = np.sqrt(metrics.mean_squared_error(actual_prices, predicted_prices))

  # Mean Absolute Percentage Error in degrees
  errors = abs(actual_prices - predicted_prices)
  MAPE = 100 * (errors / actual_prices)

  # Model Accuracy
  Accuracy = round(100 - np.mean(MAPE), 2)
  return (Accuracy, MAE, MSE, RMSE)


def Stock_Forecasting(model, test_data, no_days=30, n_steps=100):
  # taking the previous 100 days data for prediction
  test_data_len = len(test_data)-100
  x_input = test_data[test_data_len:].reshape(1, -1)
  temp_input = list(x_input)
  temp_input = temp_input[0].tolist()
  #  Prediction for next <no_days> days
  lst_output = []
  i = 0
  while(i < no_days):
    if (len(temp_input) > n_steps):
      # if length becomes greater than n_steps due to addition of predicted value to list
      # so now we need 100 values after the 1st one
      x_input = np.array(temp_input[1:])
      x_input = x_input.reshape(1, -1)
      x_input = x_input.reshape(1, n_steps, 1)
      yhat = model.predict(x_input, verbose=0)
      temp_input.extend(yhat[0].tolist())
      temp_input = temp_input[1:]
      lst_output.extend(yhat.tolist())
      i = i+1
    else:
      # reshaping the input data
      x_input = x_input.reshape((1, n_steps, 1))
      # using model to predict values for input data
      yhat = model.predict(x_input, verbose=0)
      temp_input.extend(yhat[0].tolist())
      # adding the future predicted values to previous lstm output
      lst_output.extend(yhat.tolist())
      i = i+1
  return (lst_output)


def plotPredictionResults(data, train_predict, test_predict, scaler, company, look_back=100):
  # Set plot size
  #rcParams['axes.facecolor'] = 'black'
  rcParams['figure.figsize'] = 30, 10
  # shift train predictions for plotting
  trainPredictPlot = np.empty_like(data)
  trainPredictPlot[:, :] = np.nan
  trainPredictPlot[look_back: len(train_predict)+look_back, :] = train_predict

  # shift test predictions for plotting
  testPredictPlot = np.empty_like(data)
  testPredictPlot[:, :] = np.nan
  testPredictPlot[len(train_predict)+(look_back*2) -
                  1: len(data)-1, :] = test_predict

  # plot baseline and predictions
  # green = predicted test data
  # blue = complete dataset
  # orange = predicted train data
  plt.plot(scaler.inverse_transform(data),
           color='blue', label='Actual Stock Price')
  plt.plot(trainPredictPlot, color='green', label='Training Predictions')
  plt.plot(testPredictPlot, color='red', label='Test Predictions')

  plt.grid(which='major', color='#cccccc', alpha=0.5)
  plt.legend(shadow=True)
  plt.title('Stock Price Prediction Graph', fontsize=15)
  plt.xlabel('Timeline', fontsize=12)
  plt.ylabel('Stock Price Value', fontsize=12)
  plt.xticks(rotation=45, fontsize=10)

  graph_name = company+"_Stock_Prediction.png"
  plt.savefig("model_images/"+graph_name)
  plt.show()

# Stock Prediction Driver


def main():
  # setting start and end date for stock data retrieval
  start_date = date(2010, 1, 1)
  end_date = date.today()
  company = "AAPL"

  # fetching data using ticker
  df = get_data(start_date, end_date, company)
  print(df)

  # cleaning fetched data / preparing for model
  data = data_cleaning(df)
  print(data)

  # Scaling data for LSTM model
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
  d = scaled_data
  print(d)

  # Splitting train and test data using time_step
  time_step = 100
  split_size = 0.65
  train_data, test_data, X_train, y_train, X_test, y_test = split_train_test(
      scaled_data, time_step, split_size)
  print(len(X_train), len(y_train), len(X_test), len(y_test))

  # reshaping data into 3D form so that it can be input to LSTM model
  X_train, X_test = reshape_data(X_train, X_test)

  # Training the LSTM model
  hist, model = LSTM_Model(X_train, y_train)
  #print("Model Loss = ",model_score(model,X_test,y_test))

  # Plotting the loss curve during model training
  plot_training_loss(hist)

  model_name = "saved_model/"+company+'_stock_predict_LSTM'

  # Saving the model for reuse
  model.save(model_name)

  # Loading the model
  model = tf.keras.models.load_model(model_name)

  # Using the trained model for prediction and check the performance metrics
  train_predict = model.predict(X_train)
  test_predict = model.predict(X_test)

  train_predict2 = scaler.inverse_transform(train_predict)
  test_predict2 = scaler.inverse_transform(test_predict)

  # Constructing df_compare for visualizing Stock prediction
  df_compare = data[len(train_data)+time_step:]
  df_compare['Predicted Close'] = test_predict2
  df_compare['Date'] = df_compare.index
  print(df_compare)
  pd.options.plotting.backend = "plotly"
  pred_fig = df_compare.plot(x='Date', y=['Close', 'Predicted Close'],
                             title='Stock Price Prediction', template='plotly_dark', kind='line')
  pred_graph = company+"_test_prediction.png"
  pred_fig.write_image("model_images/"+pred_graph, width=1200, height=800)

  pred_fig2 = df_compare.plot(x='Date', y=['Close', 'Predicted Close'],
                              title='Stock Price Prediction', template='plotly_dark', kind='scatter')
  pred_graph2 = company+"_test_prediction2.png"
  pred_fig2.write_image("model_images/"+pred_graph2, width=1200, height=800)
  #plotResults(df_compare)
  df_compare.to_csv("dataframes/"+company+"_stock_prediction.csv")

  look_back = 100
  # d = scaled data
  plotPredictionResults(d, train_predict2, test_predict2,
                        scaler, company, look_back)

  # Model Evaluation and Performance Measures and Scores
  actual_prices = df_compare['Close']
  predicted_prices = df_compare['Predicted Close']

  Accuracy, MAE, MSE, RMSE = Model_Evaluation(actual_prices, predicted_prices)
  print("\n-----Model Evaluation-----------------------------------------------------\n")
  print("LSTM Model Loss = ", model_loss(model, X_test, y_test))
  print("Model Accuracy = ", Accuracy)
  print("Mean Absolute Error = ", MAE, " degrees")
  print("Mean Squared Error = ", MSE)
  print("Root Mean Squared Error = ", RMSE)
  print("\n--------------------------------------------------------------------------\n")

  #  Future Stock Close Price Forecasting using Trained model
  # The number of days for which Stock Close price will be forecasted
  no_days = 30
  # The number of previous days used to forecast the Stock prices
  n_steps = 100
  # The array that stores forecasted values
  lst_output = Stock_Forecasting(model, test_data, no_days, n_steps)
  forecasted_values = scaler.inverse_transform(lst_output)
  next_date = df_compare['Date'][len(df_compare)-1]
  forecast_dates = []
  forecast_prices = []

  print("\n-----Stock Forecasting-----------------------------------------------------\n")
  for i in range(no_days):
    next_date += timedelta(days=1)
    forecast_dates.append(next_date)
    forecast_prices.append(forecasted_values[i][0])
    print("Day - ", i+1, " : ", next_date, " : ", forecasted_values[i][0])
  print("\n--------------------------------------------------------------------------\n")

  day_new = np.arange(1, time_step+1)
  day_pred = np.arange(time_step+1, time_step+1+no_days)

  # Forecast graph display
  df_len = len(d)-time_step
  d2 = d.tolist()
  d2.extend(lst_output)

  #rcParams['axes.facecolor'] = 'black'
  plt.plot(day_pred, forecasted_values, color='red', label='forecasted data')
  plt.plot(scaler.inverse_transform(
      d2[df_len:]), color='blue', label='historical data')
  plt.grid(which='major', color='#cccccc', alpha=0.5)
  plt.legend(shadow=True)
  plt.title('Stock Price Prediction Graph', fontsize=15)
  plt.xlabel('Timeline', fontsize=12)
  plt.ylabel('Stock Price Value', fontsize=12)
  plt.xticks(rotation=45, fontsize=10)
  forecast_name = company+"_Stock_Forecast.png"
  plt.savefig("model_images/"+forecast_name)
  plt.show()

  fdates = np.array(forecast_dates)
  fprices = np.array(forecast_prices)
  forecast_df = pd.DataFrame({'Date': fdates, 'Close Price': fprices},
                             columns=['Date', 'Close Price'])
  print(forecast_df)
  fname = company+"_stock_forecast.csv"
  forecast_df.to_csv('dataframes/'+fname)


main()
