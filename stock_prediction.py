# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021 (v1); 19/07/2021 (v2); 25/07/2023 (v3)
# Date: 26/08/2023 (v4) by Gia Huy Huynh

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, InputLayer
from plotly.subplots import make_subplots

#------------------------------------------------------------------------------
# Parameters
DATA_SOURCE = "yahoo"
COMPANY     = "TSLA"
PRICE_VALUE = "Close"

TRAIN_START = '2012-01-01'
TRAIN_END   = '2017-12-31'

TEST_START = "2018-01-01"
TEST_END   = "2022-12-31"

TIME_WINDOW = 7

# Number of days to look back to base the prediction
PREDICTION_DAYS = 60

EPOCH = 15
BATCH = 64

#------------------------------------------------------------------------------
# Load Data
## TO DO:
## 2) Use a different price value eg. mid-point of Open & Close
## 3) Change the Prediction days
def load_data(ticker: str="TSLA", source: str="yahoo",
              start: str=TRAIN_START, end: str=TEST_END, col: str="Close",
              split_by: str="random", split_pt: float or str=None,
              store_data: bool=True, store_scaler: bool=True):
    """
    Load data from Yahoo Finance source (for now) with pre-processing, scaling, normalising, and splitting
    Params:
        ticker       (str)          : The ticker to load (e.g. AMZN), default to TSLA
        start, end   (str)          : The start and end dates (training and testing inclusive)
        col          (str)          : The column/feature/price value to feed into the model
        split_by     (str)          : The method of data splitting - date, ratio, or random, default to ratio
        split_pt     (float, str)   : The point of data splitting, e.g. ratio=0.6 (60/40 training and testing) or date="2021-01-01" (specific date)
        store_data   (bool)         : To store data locally or not
        store_scaler (bool)         : To store scalers locally or not
    
    Returns:
        data                : The original dataframe
        x_train, y_train    : Training dataset
        x_test,  y_test     : Testing dataset
        scaler              : The scaler for inverse transformation
        dates               : The test dates
    """

    # For more details: 
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html
    #------------------------------------------------------------------------------
    # Prepare Data
    # 1) Check if data has been prepared before. 
    # If so, load the saved data
    # If not, save the data as CSV file to the data directory
    if not os.path.isdir("data"):
        os.mkdir("data")

    data_file = os.path.join(f"data/{ticker}_{start}_{end}.csv")

    # If the CSV file is found locally
    if os.path.isfile(data_file):
        # Read the CSV data with index column being the Date
        data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    else:
        data = yf.download(ticker, start, end)
        if (store_data):
            data.to_csv(data_file)

    # Handle NaN
    data.dropna()

    scaler_file = os.path.join("data/scaler.save")

    if os.path.isfile(scaler_file):
        scaler = joblib.load(scaler_file)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Note that, by default, feature_range=(0, 1). Thus, if you want a different 
        # feature_range (min, max) then you'll need to specify it here
        if store_scaler:
            joblib.dump(scaler, scaler_file)

    scaled_data = scaler.fit_transform(data[col].values.reshape(-1, 1)) 
    scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array
    # Flatten and normalise the data
    # First, we reshape a 1D array(n) to 2D array(n,1)
    # We have to do that because sklearn.preprocessing.fit_transform()
    # requires a 2D array
    # Here n == len(scaled_data)
    # Then, we scale the whole array to the range (0,1)
    # The parameter -1 allows (np.)reshape to figure out the array size n automatically 
    # values.reshape(-1, 1) 
    # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
    # When reshaping an array, the new shape must contain the same number of elements 
    # as the old shape, meaning the products of the two shapes' dimensions must be equal. 
    # When using a -1, the dimension corresponding to the -1 will be the product of 
    # the dimensions of the original array divided by the product of the dimensions 
    # given to reshape so as to maintain the same number of elements.

    x_data = []
    y_data = []

    for x in range(PREDICTION_DAYS, len(scaled_data)):
        x_data.append(scaled_data[x - PREDICTION_DAYS:x])
        y_data.append(scaled_data[x])

    # Convert them into an array
    x_data, y_data = np.array(x_data), np.array(y_data)
    # Now, x_data is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
    # and q = PREDICTION_DAYS; while y_data is a 1D array(p)

    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    # We now reshape x_data into a 3D array(p, q, 1); Note that x_train 
    # is an array of p inputs with each input being a 2D array 

    # Split the Data by Ratio, Date, or Randomly
    # Define the test dates according to the splitting option
    if split_by == "ratio":
        split_idx = int(split_pt * len(x_data))
        x_train, x_test = x_data[:split_idx], x_data[split_idx:]
        y_train, y_test = y_data[:split_idx], y_data[split_idx:]
        dates = data.iloc[split_idx + PREDICTION_DAYS:].index

    elif split_by == "date":
        # Use asof to get approximate date
        # e.g. if the given date is not reported in the DataFrame, get the closest date
        split_idx = data.index.get_loc(data.index.asof(split_pt)) - PREDICTION_DAYS
        x_train, x_test = x_data[:split_idx], x_data[split_idx:len(data) - PREDICTION_DAYS]
        y_train, y_test = y_data[:split_idx], y_data[split_idx:len(data) - PREDICTION_DAYS]
        dates = data.iloc[split_idx + PREDICTION_DAYS:].index

    else:
        # The test_idx return value is the indices of the split, keeping track of the appropriate dates from the original dataset
        x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(x_data, y_data, range(len(x_data)))
        dates = data.iloc[np.array(test_idx) + PREDICTION_DAYS].index

    return data, x_train, y_train, x_test, y_test, scaler, dates

#------------------------------------------------------------------------------
# Plot Data using Candlestick Chart
def plot_candlestick(data: pd.DataFrame, title: str, days: int=1):
    """
    Visualise historical data for analysis using Candlestick chart with the option to adjust the moving window of n trading days (n >= 1)
    Params:
        data    (pandas.DataFrame)  : The historical data to visualise
        title   (str)               : The title of the chart - to be consistent with Datetime, Company, etc.
        days    (int)               : The number of trading days for each candlestick to represent
    """
    
    if days > 1:
        # Calculate the mean of each attribute over n trading days
        # Aggregate n trading days into one data
        data = data.resample(f'{days}D').agg({
            'Open': 'mean',
            'High': 'mean',
            'Low': 'mean',
            'Close': 'mean',
            'Volume': 'sum'
        }).dropna()

        # Alternative: Each attribute can be calculated by their characteristics
        # e.g
        # - Open  - opening price of the first trading day in the range,
        # - High  - highest price within the date range
        # - Low   - lowest price within the date range
        # - Close - closing price of the last trading day in the range
        
        #   'Open' : 'first',
        #   'High' : 'max',
        #   'Low'  : 'min',
        #   'Close': 'last'

    # Store start and end dates of each time window in a series
    # For clarity purposes only
    windows = data.index.to_series().apply(
        lambda x: f"{x.strftime('%Y-%m-%d')} to {(x + pd.Timedelta(days=days-1)).strftime('%Y-%m-%d')}")

    # Create a Candlestick chart
    candlestick = go.Candlestick(
        x=windows,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        showlegend=False,
    )

    volume = go.Bar(
        x=windows,
        y=data['Volume'],
        marker={
            "color": "lightgrey",
        },
        showlegend=False,
    )
    
    # fig = go.Figure(data=[candlestick])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('OHLC', 'Volume'),
        row_width=[0.3, 0.7]
    )

    fig.add_trace(candlestick, row=1, col=1)
    fig.add_trace(volume, row=2, col=1) 

    # Add labels to the chart
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
    )

    # Hide the slide bar
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()

#------------------------------------------------------------------------------
# Plot Data using Boxplot Chart
def plot_box(data: pd.DataFrame, title: str, col: str='Close', days: int=30):

    """
    Visualise historical data for analysis using Box plot over a moving window of n trading days (n >= 1)
    Params:
        data    (pandas.DataFrame)  : The historical data to visualise
        title   (str)               : The title of the chart - to be consistent with Datetime, Company, etc.
        col     (str)               : The column/feature to plot (e.g. Open, High, Low, Close, Volume)
        days    (int)               : The number of trading days for each box to represent
    """

    # Use groupby() function to group DataFrame using a particular key or logic
    # level=0 specifies grouping by DataFrame's index, which is the Date column
    windows = data.groupby(pd.Grouper(level=0, freq=f"{days}D"))
    
    boxes = []

    for time, window in windows:
        # If no data in the window, skip it
        if len(window) == 0:
            continue

        # Specify the start and end dates of each time window
        start = window.index.min().date()
        end   = window.index.max().date()

        # Create a boxplot for each time window
        box = go.Box(
            y = window[col],
            name = f"{start} to {end}"
        )

        boxes.append(box)

    layout = go.Layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=f"{col} Prices",
        boxmode='group'
    )

    fig = go.Figure(data=boxes, layout=layout)
    fig.show()

#------------------------------------------------------------------------------
## MAIN ##
#------------------------------------------------------------------------------
## split_by = "random" - Default
# data, x_train, y_train, x_test, y_test, scaler, dates = load_data()

## split_by = "date"
# data, x_train, y_train, x_test, y_test, scaler, dates = load_data(split_by="date", split_pt=TEST_START)

## split_by = "ratio"
data, x_train, y_train, x_test, y_test, scaler, dates = load_data(split_by="ratio", split_pt=0.5)
print(data)

# The following lines are to scale the historical data to fit the testing frame
dt_range = pd.date_range(start="2022-01-01", end="2022-03-01")
data = data[data.index.isin(dt_range)]

# Plot historical data using Candlestick chart
plot_candlestick(data, f"{COMPANY} Share Price", TIME_WINDOW)

# Plot historical data using Box plot
plot_box(data, f"{COMPANY} Share Price in {TIME_WINDOW}-Day Windows", "Open", TIME_WINDOW)

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=EPOCH, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
actual_prices    = scaler.inverse_transform(y_test.reshape(-1, 1))

#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

# Refine test dates to only YEAR for plotting
dates = pd.to_datetime(dates)
years = dates.year.unique()

plt.plot(dates, actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(dates, predicted_prices, color="green", label=f"Predicted {COMPANY} Price")

plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------
model_inputs = data[len(data) - len(y_test) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the
# data from the training period
model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line
real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day prediction,
# it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN
# to analyse the images of the stock price changes to detect some patterns with the trend of the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??