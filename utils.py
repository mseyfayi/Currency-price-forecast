import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dropout


def split_sequence(seq, n_steps_in, n_steps_out):
    """
    Splits the multivariate time sequence
    """

    # Creating a list for both variables
    X, y = [], []

    for i in range(len(seq)):

        # Finding the end of the current sequence
        end = i + n_steps_in
        out_end = end + n_steps_out

        # Breaking out of the loop if we have exceeded the dataset's length
        if out_end > len(seq):
            break

        # Splitting the sequences into: X = past prices and indicators, y = prices ahead
        seq_x, seq_y = seq[i:end, :], seq[end:out_end, 0]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def visualize_training_results(results):
    """
    Plots the loss and accuracy for the training and testing data
    """
    history = results.history
    plt.figure(figsize=(16, 5))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    plt.figure(figsize=(16, 5))
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


def layer_maker(n_layers, n_nodes, activation, model, drop=None, d_rate=.5):
    """
    Creates a specified number of hidden layers for an RNN
    Optional: Adds regularization option - the dropout layer to prevent potential overfitting (if necessary)
    """

    # Creating the specified number of hidden layers with the specified number of nodes
    try:

        for x in range(1, n_layers + 1):
            model.add(LSTM(n_nodes, activation=activation, return_sequences=True))

            # Adds a Dropout layer after every Nth hidden layer (the 'drop' variable)
            if x % drop == 0:
                model.add(Dropout(d_rate))
    except:
        pass


def validater(n_per_in, n_per_out, n_features, close_scaler, df, model):
    """
    Runs a 'For' loop to iterate through the length of the DF and create predicted values for every stated interval
    Returns a DF containing the predicted values for the model with the corresponding index values based on a business day frequency
    """

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
                               index=pd.date_range(start=x.index[-1],
                                                   periods=len(yhat),
                                                   freq="B"),
                               columns=[x.columns[0]])

        # Updating the predictions DF
        predictions.update(pred_df)

    return predictions


def val_rmse(df1, df2):
    """
    Calculates the root mean square error between the two Dataframes
    """
    df = df1.copy()

    # Adding a new column with the closing prices from the second DF
    df['close2'] = df2.Close

    # Dropping the NaN values
    df.dropna(inplace=True)

    # Adding another column containing the difference between the two DFs' closing prices
    df['diff'] = df.Close - df.close2

    # Squaring the difference and getting the mean
    rms = (df[['diff']] ** 2).mean()

    # Returning the sqaure root of the root mean square
    return float(np.sqrt(rms))
