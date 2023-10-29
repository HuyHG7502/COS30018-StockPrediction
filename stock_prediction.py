# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021 (v1); 19/07/2021 (v2); 25/07/2023 (v3)
# Date: 26/09/2023 (v4) by Gia Huy Huynh

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
import pmdarima as pm

import os
import joblib
import random

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional, Reshape
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from plotly.subplots import make_subplots

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import adfuller

#------------------------------------------------------------------------------
# Parameters
DATA_SOURCE = "yahoo"
TICKER      = "TSLA"

FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
FEATURE  = "Close"

TRAIN_START = '2010-01-01'
TRAIN_END   = '2017-12-31'

TEST_START = "2018-01-01"
TEST_END   = "2023-01-10"

WINDOW = 30

# Number of days to look back to base the prediction
N_LOOKUP = 120
N_STEPS  = 7

EPOCH = 50
BATCH = 50

CELL = GRU
LAYERS = [50, 50]
DROPOUT = 0.2
BIDIRECTIONAL = True

LOSS = "huber_loss"
OPTIMIZER = "adam"

DATA_DIR = "data"
MODEL_DIR = "model"
RESULT_DIR = "result"

ORDER = (5, 1, 1)

N_ESTIMATORS = 100
N_RANDOM = 75

#------------------------------------------------------------------------------
# Set seed so the same results are achieved for several runs 
np.random.seed(75)
tf.random.set_seed(75)
random.seed(75)

#------------------------------------------------------------------------------
# Load and Process Data
def load_data(ticker=TICKER, start=TRAIN_START, end=TEST_END, store_data=True):
    """
    Load data from Yahoo Finance source.
    Params:
        ticker       (str, pd.DataFrame) : The ticker to load (e.g. AMZN) or the loaded data
        start, end   (str)               : The start and end dates for the data
        store_data   (bool)              : To store data locally or not

    Returns:
        df          (pd.DataFrame)      : The loaded dataframe
    """

    # Create the data directory if it doesn't already exist
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    # Check if the ticker is already a loaded stock from Yahoo Finance
    # If yes, use it directly as data
    if isinstance(ticker, pd.DataFrame):
        df = ticker
    # If not, load it from yfinance or local CSV
    elif isinstance(ticker, str):
        data_file = os.path.join(DATA_DIR, f"{ticker}_{start}_{end}.csv")

        # If the CSV file is found locally
        if os.path.isfile(data_file):
            # Read the CSV data with index column being the Date
            df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        else:
            df = yf.download(ticker, start, end)
    else:
        raise TypeError("ticker must be either a `str` or a `pd.DataFrame` instance.")

    # Handle NaN
    df.dropna(inplace=True)

    # Store data as local CSV
    if store_data:
        df.to_csv(data_file)
    
    return df

def prepare_data(df, cols=FEATURES,
                 n_lookup=N_LOOKUP, n_steps=N_STEPS,
                 split_by="random", split_pt=None, store_scaler=True):
    """
    Pre-process the data: scaling, normalising, and splitting
    Params:
        df           (pd.DataFrame)  : The dataframe to be processed
        cols         (list)          : The column features to feed into the model
        n_lookup     (int)           : The days in the past to base prediction on (historical sequence length)
        n_steps      (int)           : The days in the future to make prediction for
        split_by     (str)           : The method of data splitting - date, ratio, or random, default to ratio
        split_pt     (float, str)    : The point of data splitting, e.g. ratio=0.6 (60/40 training and testing) or date="2021-01-01" (specific date)
        store_data   (bool)          : To store data locally or not

    Returns:
        result       (dict)          : The resulting dictionary containing various components

    """

    for col in cols:
        assert col in df.columns, f'Invalid {col} column in the dataframe.'

    assert split_by in ['random', 'ratio', 'date'], f"Invalid split method. Expected 'random', 'ratio', 'date'."
    
    if split_by == "ratio":
        split_idx = int(split_pt * len(df))

        df_train = df[:split_idx]
        df_test  = df[split_idx:]

    elif split_by == "date":
        df_train = df[df.index < split_pt]
        df_test  = df[df.index > split_pt]

    else:
        df_train, df_test  = train_test_split(df)

    data = df.copy()
    scalers = {}

    for col in cols:
        scaler_file = os.path.join(DATA_DIR, f"scaler_{col}.save")

        if os.path.isfile(scaler_file):
            scaler = joblib.load(scaler_file)
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            # Note that, by default, feature_range=(0, 1). Thus, if you want a different 
            # feature_range (min, max) then you'll need to specify it here
            if store_scaler:
                joblib.dump(scaler, scaler_file)

        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
        scalers[col] = scaler

    scaled_data = data[cols].values

    #------------------------------------------------------------------------------
    x_train, y_train, d_train = [], [], []

    for i in range(n_lookup, len(df_train) - n_steps + 1):
        x_train.append(scaled_data[i - n_lookup: i])
        y_train.append(scaled_data[i:i + n_steps])
        d_train.append(df.index[i])

    x_test, y_test, d_test = [], [], []
    
    for i in range(len(df_train), len(df) - n_steps + 1):
        x_test.append(scaled_data[i - n_lookup: i])
        y_test.append(scaled_data[i:i + n_steps])
        d_test.append(df.index[i])

         
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test,  y_test  = np.array(x_test),  np.array(y_test)

    model_inputs = x_test[-1]
    model_inputs = np.reshape(model_inputs, (1, n_lookup, len(cols)))

    return {
        'df'           : df.copy(),
        'scalers'      : scalers,
        'df_train'     : df_train,
        'df_test'      : df_test,
        'x_train'      : x_train,
        'y_train'      : y_train,
        'x_test'       : x_test,
        'y_test'       : y_test,
        'd_train'      : d_train,
        'd_test'       : d_test,
        'model_inputs' : model_inputs
    }

#------------------------------------------------------------------------------
# Create AI Models for Prediction
def create_neural_model(cell=CELL, layers=LAYERS,
                        n_steps=N_STEPS, n_lookup=N_LOOKUP, n_features=len(FEATURES),
                        dropout=DROPOUT, loss=LOSS, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL):
    """
    Create a Neural Network / Deep Learning Model based on given parameters
    Params:
        cell            (LSTM | GRU ! SimpleRNN)   : The type of RNN cell to train the model          - Default is LSTM
        units           (int)                      : The number of units (neurons) for each RNN layer - Default is 100
        n_features      (int)                      : The number of features in the input data         - Default is 1
        n_layers        (int)                      : The number of RNN layers in the model            - Default is 2
        n_lookup        (int)                      : The number of time steps in each input sequence
        dropout         (float)                    : The dropout rate for regularisation to prevent overfitting
        loss            (str)                      : The loss function to be used during training
        optimizer       (str)                      : The optimisation algorithm to be used during training
        bidirectional   (bool)                     : Whether to use bidirectional RNN layers
    """
    
    model = Sequential()
    for layer, size in enumerate(layers):
        # Determine if this is the first, last, or hidden layers
        is_first = layer == 0
        is_last  = layer == len(layers) - 1

        if bidirectional:
            if is_first:
                # Specify the input shape for the first layer
                model.add(Bidirectional(cell(size, return_sequences=True, input_shape=(n_lookup, n_features))))
            else:
                # Return sequences for all layers except the last
                model.add(Bidirectional(cell(size, return_sequences=(not is_last))))
        else:
            if is_first:
                model.add(cell(size, return_sequences=True, input_shape=(n_lookup, n_features)))
            else:
                model.add(cell(size, return_sequences=(not is_last)))
        
        # Add Dropout after each layer for regularisation
        model.add(Dropout(dropout))

    # Default Output layer:
    model.add(Dense(n_steps * n_features, activation="linear"))
    model.add(Reshape((n_steps, n_features)))
    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=["mean_absolute_error"])

    return model

def train_neural_model(data):
    """
    Traing and deploy the Neural Network model
    """
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    model_file = os.path.join(MODEL_DIR, f"{CELL.__name__}_{LOSS}_{OPTIMIZER}_seq-{N_LOOKUP}_steps-{N_STEPS}_layers-{len(LAYERS)}")

    # If the model already exists in local directory
    # Load it
    if os.path.isfile(f"{model_file}.h5"):
        model = load_model(f"{model_file}.h5")
    # Else, create a new one
    else:
        model = create_neural_model()
        # optimizer='rmsprop'/'sgd'/'adadelta'/...
        # loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...
        
        # reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, min_lr=0.0001)
        checkpointer = ModelCheckpoint(f"{model_file}-best.h5",
                                    save_weights_only=True, save_best_only=True,
                                    monitor="val_loss", mode="min")
        
        # Train the model with the training data
        model.fit(data['x_train'], data['y_train'],
            epochs=EPOCH, batch_size=BATCH,
            validation_data=(data["x_test"], data["y_test"]),
            callbacks=[checkpointer])
        
        # Fine tune the model
        model = tune_neural_model(model, model_file, data['x_test'], data['y_test'])

    return model

def tune_neural_model(model, path, x_test, y_test):
    """
    Refine the model's weight for optimisation
    """
    model.save_weights(f"{path}-test.h5")
    loss_test, _ = model.evaluate(x_test, y_test)

    model.load_weights(f"{path}-best.h5")
    loss_best, _ = model.evaluate(x_test, y_test)

    if (loss_best > loss_test):
        model.load_weights(f"{path}-test.h5")
    
    model.save(f"{path}.h5")

    # plot_model(model, to_file=f"{model_file}.png", show_shapes=True, show_layer_names=True)

    return model    

def train_arima_model(data, order=ORDER):
    """
    Create and train a ARIMA/SARIMA model based on the given order (can be later computed)
    """
    # If order is not specified - use auto_arima for auto-computation
    # order = pm.auto_arima(data['df_train']['Close'], seasonal=False, stepwise=True)
    # print(order.order)
    
    # Create an ARIMA model with the training dataset for 'Close' feature
    model = ARIMA(data['df_train']['Close'], order=order).fit()

    return model

def train_rf_model(data):
    """
    Create and train a Random Forest model
    """
    # Unlike LSTM, the input shape of the RF model is 2D
    # So we need to reshape the x_train and y_train data
    x_train = data['x_train'].reshape((data['x_train'].shape[0], -1))
    y_train = data['y_train'].reshape((data['y_train'].shape[0], -1))

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=N_RANDOM)
    model.fit(x_train, y_train)

    return model

#------------------------------------------------------------------------------
# Predict Next Day
def predict_neural(model, data, dates, scalers, cols=FEATURES):
    # Predicting the stock prices using the model
    pred = model.predict(data)

    # Inverse transforming the normalized predictions
    predictions = {}
    for i, col in enumerate(cols):
        scaler = scalers[col]
        # Reshape the array to 2D before inverse transforming, then flatten back to 1D
        # Extract only the first day's prediction for each input sample
        predictions[col] = scaler.inverse_transform(pred[:,:,i].reshape(-1, 1)).flatten()
    
    # Creating a DataFrame from the inverse-transformed predictions
    pred_df = pd.DataFrame(predictions)
    
    # Generate a new index for the DataFrame
    dates_idx = []
    start_idx = len(dates) - len(data)
    for i in range(len(data)):
        current_date = dates[start_idx + i]
        for step in range(N_STEPS):
            next_date = current_date + timedelta(days=step)
            while next_date not in dates and next_date <= dates[-1]:
                # Increment the date until a matching date is found or the end of the date range is reached
                next_date += timedelta(days=1)
            dates_idx.append(next_date)

    # Assign the new index to the DataFrame
    pred_df.index = pd.DatetimeIndex(dates_idx)
    pred_df = pred_df.groupby(pred_df.index).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Adj Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return pred_df

def predict_arima(model, data, dates):
    predictions = model.forecast(data.shape[0])

    pred_df = pd.DataFrame({FEATURE: predictions.values}, index=dates)
    for col in FEATURES:
        if col != FEATURE:
            pred_df[col] = data[col]

    return pred_df

def predict_rf(model, data, dates, scalers, cols=FEATURES):
    pred = model.predict(data)

    # Inverse transforming the normalized predictions
    predictions = {}
    for i, col in enumerate(cols):
        scaler = scalers[col]
        # Reshape the array to 2D before inverse transforming, then flatten back to 1D
        # Extract only the first day's prediction for each input sample
        predictions[col] = scaler.inverse_transform(pred[:,:,i].reshape(-1, 1)).flatten()
    
    # Creating a DataFrame from the inverse-transformed predictions
    pred_df = pd.DataFrame(predictions, index=dates)

    return pred_df

def predict_ensemble(lstm_pred, arima_pred):
    ensemble_pred = []
    for i in range(len(lstm_pred)):
        ensemble_pred.append((lstm_pred.iloc[i][FEATURE] + arima_pred.iloc[i][FEATURE]) / 2)

    return ensemble_pred

### Legacy Code
### For Rolling Forecasts
def predict_rolling_forecasts(model, data):
    predictions = []
    
    model_inputs = data['df'][len(data['df']) - len(data['y_test']) - N_LOOKUP:].copy()
    # Scale the input data
    for feature in FEATURES:
            model_inputs.loc[:, feature] = data['scalers'][feature].transform(model_inputs[feature].values.reshape(-1, 1))

    for i in range(N_STEPS):
        # Extract the sequence for prediction
        model_input = model_inputs.iloc[-N_LOOKUP:].values
        model_input = np.expand_dims(model_input, axis=0)

        # Make a prediction for the next day
        prediction = model.predict(model_input)

        # Create a new DataFrame entry for the predicted day
        entry = pd.DataFrame(columns=FEATURES, index=[model_inputs.index[-1] + pd.DateOffset(1)])
        # Use the predicted value if the feature is being predicted, otherwise duplicate the last day's value
        for feature in FEATURES:
            entry[feature] = prediction[0][0] if feature == FEATURE else model_inputs.iloc[-1][FEATURE]

        # Concatenate the new entry to the original DataFrame
        model_inputs = pd.concat([model_inputs, entry])

        # Inverse transform the prediction and append it to the predictions list
        prediction = data['scalers'][FEATURE].inverse_transform(prediction)
        predictions.append(prediction[0][0])

    for i, prediction in enumerate(predictions):
        print(f"Prediction Day {i+1}:\t{prediction:.2f}")

#------------------------------------------------------------------------------
# Store Figures as Images
def export_fig(fig, name):
    if not os.path.isdir("imgs"):
        os.mkdir("imgs")

    fig.write_image(f"imgs/{name}.png")

def resample_data(data, days):
    assert days >= 0, "Days must be >= 1."

    aggregation = {
          'Open': 'first',
          'High': 'max',
          'Low': 'min',
          'Close': 'last',
          'Adj Close': 'last',
          'Volume': 'sum'
    }

    return data.resample(f'{days}d').agg(aggregation).dropna()

#------------------------------------------------------------------------------
# Plot the Graph
def plot(data, pred, dates):

    plt.figure(figsize=(15, 6))
    plt.plot(dates, data, color="blue", label=f"Actual {TICKER} Price")
    plt.plot(dates, pred, color="green", label=f"Predicted {TICKER} Price")

    plt.title(f"{TICKER} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{TICKER} Share Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------
# Plot Data using Candlestick Chart
def plot_candlestick(data, pred, days=1, col='Close'):
    """
    Visualise historical data for analysis using Candlestick chart with the option to adjust the moving window of n trading days (n >= 1)
    Params:
        data    (pd.DataFrame)  : The historical data to visualise
        title   (str)           : The title of the chart - to be consistent with Datetime, TICKER, etc.
        days    (int)           : The number of trading days for each candlestick to represent
    """

    # Calculate the mean of each attribute over n trading days
    # Aggregate n trading days into one data
    data = resample_data(data, days)
    pred = resample_data(pred, days)

        # Alternative: Each attribute can be calculated by their characteristics
        # e.g
        # - Open  - opening price of the first trading day in the range,
        # - High  - highest price within the date range
        # - Low   - lowest price within the date range
        # - Close - closing price of the last trading day in the range
        
        #   'Open': 'first',
        #   'High': 'max',
        #   'Low': 'min',
        #   'Close': 'last',
        #   'Adj Close': 'last',
        #   'Volume': 'sum'

    # Create a Candlestick chart
    candlestick = go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC',
        showlegend=False,
    )

    volume = go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        showlegend=False,
    )

    pred = go.Scatter(
        x=pred.index,
        y=pred[col],
        marker={"color": "black"},
        name=f'Predicted {col}',
    )

    real = go.Scatter(
        x=data.index,
        y=data[col],
        marker={"color": "blue"},
        name=f'Actual {col}',
    ) 
    
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.8, 0.2]
    )

    fig.add_trace(candlestick, row=1, col=1)
    fig.add_trace(volume, row=2, col=1)
    fig.add_trace(pred, row=1, col=1)
    fig.add_trace(real, row=1, col=1)

    # Add labels to the chart
    fig.update_layout(
        title=f'{TICKER} Share Prices {data.index[0]:%b %Y} - {data.index[-1]:%b %Y}',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    fig.show()

    # Export figure as PNG
    # export_fig(fig, f"Candlestick_{TICKER}_{days}_{title[-24:]}")

#------------------------------------------------------------------------------
# Plot Data using Boxplot Chart
def plot_box(data, pred, days=1, col='Close'):

    """
    Visualise historical data for analysis using Box plot over a moving window of n trading days (n >= 1)
    Params:
        
    """
    windows = data.groupby(pd.Grouper(level=0, freq=f"{days}D"))
    pred = resample_data(pred, days)

    boxes = []
    for time, window in windows:
        # If no data in the window, skip it
        if len(window) == 0:
            continue

        # Create a boxplot for each time window
        box = go.Box(
            y = window[col],
            name = f"{time}",
            showlegend=False
            # boxpoints="all"
        )
        boxes.append(box)

    scatter = go.Scatter(
        x=pred.index,
        y=pred[col],
        name=f'Predicted {col}'
    )

    layout = go.Layout(
		title=f'{TICKER} Share Prices {data.index[0]:%b %Y} - {data.index[-1]:%b %Y}',
        xaxis_title="Date",
        yaxis_title=f"{col} Prices",
    )

    fig = go.Figure(data=(boxes + [scatter]), layout=layout)
    fig.show()

    # Export figure as PNG
    # export_fig(fig, f"Box_{TICKER}_{days}_{title[-24:]}")
    # Export figure as PNG
    # export_fig(fig, f"Box_{TICKER}_{days}_{title[-24:]}")

#------------------------------------------------------------------------------
## MAIN ##
#------------------------------------------------------------------------------
data = load_data()
data = prepare_data(data, split_by="ratio", split_pt=0.8)

neural_model = train_neural_model(data)
arima_model  = train_arima_model(data)
# rf_model     = train_rf_model(data)

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

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
# data['df_pred_n'] = predict_neural(neural_model, data['x_test'], data['d_test'], data['scalers'])
data['df_pred_a'] = predict_arima(arima_model, data['df_test'], data['df_test'].index)

# Plot the Data
plot_candlestick(data['df_test'], data['df_pred_a'], days=7)
# plot_box(data['df_test'], data['df_pred'], days=7)

#------------------------------------------------------------------------------
# Predict next days
# dates = pd.date_range(data['df'].index[-1] + timedelta(days=1), periods=N_STEPS, name='Date')
# data['df_pred_n'] = predict_neural(neural_model, data['model_inputs'], dates, data['scalers'])
# data['df_pred_a'] = predict_arima(arima_model,  data['df_test'].tail(N_STEPS), dates)

# data['model_outputs'] = predict_ensemble(data['df_pred_n'], data['df_pred_a'])

# for i, price in enumerate(data['model_outputs']):
#     print(f"Predicted {FEATURE} Price for Day {i + 1}: ${price:.2f}")

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