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


from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

#----------------------------------------------Machine Learning Script for Stock Prediction-----------------------------------------------
  
# should return an array of forecasted values    
def getPredictions2(company):
    forecast_name = company+"_stock_forecast.csv"
    df = pd.read_csv("dataframes/"+forecast_name)
    output = df['Close Price']
    print("Dates = ",df['Date'])
    print("output = ",output)
    return (output, df['Date'].tolist())



#----------------------------------------------------------------------------------------------------------------------------------------------------

# root api direct to index.html (home page)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI

    '''
    features = [str(x) for x in request.form.values()]
    print(features)
    company = str(features[0])
    print(company)
    output,fdates = getPredictions2(company) 
    name1 = company+"_test_prediction2.png"
    name2 = company+"_Stock_Forecast.png"
    name3 = company+"_Stock_Prediction.png"
    name4 = company+"_test_prediction.png"

    return render_template('index.html', forecast_prices=output, fdates=fdates, imgname1=name1, imgname2=name2, imgname3=name3, imgname4=name4)

if __name__ == "__main__":
    app.run(debug=True)
















