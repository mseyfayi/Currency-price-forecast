import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import RobustScaler

from crawler import t_list, crawl
from utils import split_sequence, visualize_training_results, val_rmse

# %%
plt.style.use("bmh")

# %%
name = t_list[2]

crawl(name)
# %%
# Read CSV file
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'csv/%s.csv' % name)
df = pd.read_csv(filename)

# Add Volume column
df['Volume'] = 1

# Datetime conversion
df['Date'] = pd.to_datetime(df.Date)

# Setting the index
df.set_index('Date', inplace=True)

# Dropping any NaNs
df.dropna(inplace=True)

df = df.head(2050)

deep_copy = df.copy(deep=True)

# Adding all the indicators
df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

# Dropping everything else besides 'Close' and the Indicators
df.drop(['Open', 'High', 'Low', 'Volume', 'Date_fa'], axis=1, inplace=True)

# Scale fitting the close prices separately for inverse_transformations purposes later
close_scaler = RobustScaler()
close_scaler.fit(df[['Close']])

# Normalizing/Scaling the DF
scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# %%
# How many periods looking back to learn
n_per_in = 90
# How many periods to predict
n_per_out = 10
# Features
n_features = df.shape[1]
# Splitting the data into appropriate sequences
X, y = split_sequence(df.iloc[::-1].to_numpy(), n_per_in, n_per_out)

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
n_nodes = 30
model.add(LSTM(n_nodes, activation=activ, return_sequences=True))

# Final Hidden layer
model.add(LSTM(60, activation=activ))

# Output layer
model.add(Dense(n_per_out))

# Model summary
model.summary()

# Compiling the data with selected specifications
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Fitting and Training
res = model.fit(X, y, epochs=40, batch_size=128, validation_split=0.1)

visualize_training_results(res)

# %% Model Validation
# Transforming the actual values to their original price
actual = pd.DataFrame(close_scaler.inverse_transform(df[["Close"]]),
                      index=df.index,
                      columns=[df.columns[0]])

# Getting a DF of the predicted values to validate against
# Creating an empty DF to store the predictions
predictions = pd.DataFrame(index=df.index, columns=[df.columns[0]])

for i in range(n_per_in, len(df) - n_per_in, n_per_out):
    # Creating rolling intervals to predict off of
    x = df[-i - n_per_in:-i]

    # Predicting using rolling intervals
    yhat = model.predict(np.array(x).reshape(1, n_per_in, n_features))

    # Transforming values back to their normal prices
    yhat = close_scaler.inverse_transform(yhat)[0]

    # DF to store the values and append later, frequency uses business days
    pred_df = pd.DataFrame(yhat,
                           index=pd.date_range(start=x.index[0],
                                               periods=len(yhat),
                                               freq="B"),
                           columns=[x.columns[0]])

    # Updating the predictions DF
    predictions.update(pred_df)

# Printing the RMSE
print("RMSE:", val_rmse(actual, predictions))

# Plotting
plt.figure(figsize=(16, 6))

# Plotting those predictions
plt.plot(predictions, label='Predicted')

# Plotting the actual values
plt.plot(actual, label='Actual')

plt.title(f"Predicted vs Actual Closing Prices")
plt.ylabel("Price")
plt.legend()
plt.xlim('2018-05', '2020-05')
plt.show()

# %% Forecasting the Future
# Predicting off of the most recent days from the original DF
yhat = model.predict(np.array(df.head(n_per_in)[::-1]).reshape(1, n_per_in, n_features))

# Transforming the predicted values back to their original format
yhat = close_scaler.inverse_transform(yhat)[0][::-1]

# Creating a DF of the predicted prices
preds = pd.DataFrame(yhat,
                     index=pd.date_range(start=df.index[0] + timedelta(days=1),
                                         periods=len(yhat),
                                         freq="B"),
                     columns=[df.columns[0]])

# Transforming the actual values to their original price
actual = pd.DataFrame(close_scaler.inverse_transform(df[["Close"]].head(n_per_in)),
                      index=df.Close.head(n_per_in).index,
                      columns=[df.columns[0]])

# Printing the predicted prices
print(preds)

# Plotting
plt.figure(figsize=(16, 6))
plt.plot(actual, label="Actual Prices")
plt.plot(preds, label="Predicted Prices")
plt.ylabel("Price")
plt.xlabel("Dates")
plt.title(f"Forecasting the next {len(yhat)} days")
plt.legend()
plt.show()
