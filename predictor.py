# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler
import ta
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

from utils import split_sequence, layer_maker, visualize_training_results

plt.style.use("bmh")

# %% Prepare DataFrame

# Read CSV file
df = pd.read_csv('/home/mohamad/Predictor/csv/price_dollar_rl.csv')

# Add Volume column
df['Volume'] = 1

# Datetime conversion
df['Date'] = pd.to_datetime(df.Date)

# Setting the index
df.set_index('Date', inplace=True)

# Dropping any NaNs
df.dropna(inplace=True)

# %% Technical Indicators

# Adding all the indicators
df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

# Dropping everything else besides 'Close' and the Indicators
df.drop(['Open', 'High', 'Low', 'Volume', 'Date_fa'], axis=1, inplace=True)

# %% Scaling

# Scale fitting the close prices separately for inverse_transformations purposes later
close_scaler = RobustScaler()
close_scaler.fit(df[['Close']])

# Normalizing/Scaling the DF
scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# %% Splitting the Data
# How many periods looking back to learn
n_per_in = 90
# How many periods to predict
n_per_out = 30
# Features
n_features = df.shape[1]
# Splitting the data into appropriate sequences
X, y = split_sequence(df.to_numpy(), n_per_in, n_per_out)

# %% Neural Network Modeling

# Instatiating the model
model = Sequential()

# Activation
activ = "tanh"

# Input layer
model.add(LSTM(90,
               activation=activ,
               return_sequences=True,
               input_shape=(n_per_in, n_features)))

# Hidden layers
layer_maker(n_layers=1,
            n_nodes=30,
            activation=activ,
            model=model)

# Final Hidden layer
model.add(LSTM(60, activation=activ))

# Output layer
model.add(Dense(n_per_out))

# Model summary
model.summary()

# Compiling the data with selected specifications
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# %% Fitting and Training
res = model.fit(X, y, epochs=50, batch_size=128, validation_split=0.1)

# %% Visualizing Loss and Accuracy
visualize_training_results(res)
