import mlflow
import mlflow.pytorch

mlflow.set_tracking_uri(uri="http://127.0.0.1:7777")
mlflow.set_experiment("LSTM Experiments")

############################################################################
##################  SOME GENERAL PLOT  CONFIGURATIONS  #####################
############################################################################

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from tqdm.notebook import tqdm

# %matplotlib inline

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)

Colour_Palette = ['#01BEFE', '#FF7D00', '#FFDD00', '#FF006D', '#ADFF02', '#8F00FF']
sns.set_palette(sns.color_palette(Colour_Palette))

tqdm.pandas()

############################################################################
###############  YFINANCE - DOWNLOAD DATA AND CREATE DF  ###################
############################################################################

import yfinance as yf
from datetime import date
import pandas as pd
import numpy as np

stock_end_date = date.today().strftime("%Y-%m-%d") # GRUPO : DECIDIR
stock_start_date = '2020-01-01'                    # GRUPO : DECIDIR
tickers = ['PETR4.SA', 'BZ=F', '6L=F']

df_full = yf.download(tickers, start=stock_start_date, end=stock_end_date)

# Inspect the data
print(df_full.head())
print(df_full.info())

# Split into 1 sub DataFrame by ticker
n_tickers = len(tickers)
sub_df = {}
for tk in tickers:
    sub_df[tk] = df_full.xs(key=tk, level='Ticker', axis=1, drop_level=False)

############################################################################
############  FUNCTION TO PLOT YFINANCE DATA THROUGH DATES  ################
############################################################################

from plot_fn import data_plot_multindex, data_plot

# Plot the data
# data_plot_multindex(df_full,n_tickers)

# data_plot(sub_df['PETR4.SA'])
# data_plot(sub_df['CL=F'])
# data_plot(sub_df['BZ=F'])
# data_plot(sub_df['6L=F'])
# plt.show()

############################################################################
##############  SPLIT TRAIN AND TEST DATA + RESHAPE DATA  ##################
############################################################################

import math

# Selecting only Close Values for everyone
df = df_full['Close']

# Train test split

def train_test_split_fn(df, train_perc_size):

    training_data_len = math.ceil(len(df) * train_perc_size)     # GRUPO : DECIDIR
    print(training_data_len)

    # Splitting the dataset
    train_data = df[:training_data_len].iloc[:, 0:n_tickers]
    test_data = df[training_data_len:].iloc[:, :n_tickers]
    print(train_data.shape, test_data.shape)

    return train_data, test_data

train_data, test_data = train_test_split_fn(df,train_perc_size=0.8)

def reshape_to_np_array(train_data, test_data, new_size):

    # Selecting Open Price values
    dataset_train = train_data.values  # GRUPO : CLOSE
    # Reshaping 1D to 2D array
    dataset_train = np.reshape(dataset_train, (-1, new_size))

    # Selecting Open Price values
    dataset_test = test_data.values    # GRUPO : CLOSE
    # Reshaping 1D to 2D array
    dataset_test = np.reshape(dataset_test, (-1, new_size))

    return dataset_train, dataset_test

dataset_train, dataset_test =  reshape_to_np_array(train_data, test_data, n_tickers)
print(dataset_train.shape)
print(dataset_test.shape)

############################################################################
####################  SCALING DATA WITH MINMAXSCALER  ######################
############################################################################

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1),clip=True)     # GRUPO : DECIDIR
# Scaling dataset - FIT SÓ AQUI
scaled_train = scaler.fit_transform(dataset_train)
np.nan_to_num(scaled_train,copy=False,nan=0.0)
print(scaled_train[:5])

# Normalizing values between 0 and 1 - AQUI SÓ TRANSFORM
scaled_test = scaler.transform(dataset_test)
np.nan_to_num(scaled_test,copy=False,nan=0.0)
print(scaled_test[:5])

############################################################################
##############  SPLIT DATA INTO X (INPUTS) AND Y (LABLES)  #################
############################################################################

# COLOCAR ESSE STEP NA PIPELINE DE DADOS

def create_dataset_from_moving_window(scaled_data, window_length, n_features):
    
    # Create sequences and labels for training data
    L_dataset = len(scaled_data)
    X, y = [], []
    for i in range(L_dataset - window_length):
        X.append(scaled_data[i:i + window_length,0:n_features])
        y.append(scaled_data[i + window_length,n_features-1:])  # Predicting the value right after the sequence
    X, y = np.array(X), np.array(y)
    return X, y

W_train = 50
W_test = 50
X_train, y_train = create_dataset_from_moving_window(scaled_train, window_length=W_train, n_features=n_tickers)
X_test, y_test = create_dataset_from_moving_window(scaled_test, window_length=W_test, n_features=n_tickers)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

############################################################################
################  CONVERT DATA TO PYTORCH TENSOR  ##########################
############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
print(X_train.shape, y_train.shape)

# Convert data to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
print(X_test.shape, y_test.shape)

############################################################################
##############  CREATING LSTM CUSTOM CLASS WITH LINEAR OUT  ################
############################################################################

class LSTMModel(nn.Module):
    # AVALIAR INICIALIZAÇÃO DOS PESOS
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

############################################################################
######################  SETUP / CONFIGS FOR TRAINING  ######################
############################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

input_size = n_tickers          # GRUPO : HYPERPARAMETROS
num_layers = 10         # GRUPO : HYPERPARAMETROS
hidden_size = 100       # GRUPO : HYPERPARAMETROS
output_size = 1
dropout = 0.2           # Regulatization // GRUPO : HYPERPARAMETROS
learning_rate = 0.001  # GRUPO : HYPERPARAMETROS

model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
loss_fn = nn.MSELoss(reduction='mean')  # GRUPO : DECIDIR
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # GRUPO : DECIDIR

batch_size = 30  # GRUPO : VERIFICAR
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 10  # GRUPO : HYPERPARAMETROS
train_hist = []
test_hist = []

############################################################################
###################  LSTM TRAINING // LSTM TRANING  ########################
############################################################################

params_to_log = {
    "start_date" : stock_start_date,
    "end_date" : stock_end_date,
    "batch_size" : batch_size,
    "sequence_length_train" : W_train,
    "sequence_length_test" : W_test,
    "input_size" : input_size,
    "num_layers" : num_layers,
    "hidden_size" : hidden_size,
    "dropout" : dropout,
    "learning_rate" : learning_rate,
    "num_epochs" : num_epochs,
    "optimizer" : optimizer,
    "loss_fn" : loss_fn,
}


import time

start_training_time = time.time()
print(f"Starting training: {start_training_time}")

with mlflow.start_run():

    mlflow.set_tag("Fase","1 - Define Baseline")
    mlflow.log_params(params_to_log)

    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        train_hist.append(average_loss)

        model.eval()
        with torch.no_grad():
            total_test_loss = 0.0

            for batch_X_test, batch_y_test in test_loader:
                batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                predictions_test = model(batch_X_test)
                test_loss = loss_fn(predictions_test, batch_y_test)

                total_test_loss += test_loss.item()

            average_test_loss = total_test_loss / len(test_loader)
            test_hist.append(average_test_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')

    end_training_time = time.time()
    elapsed_training_time = end_training_time - start_training_time
    print(f"Ending training: {end_training_time}")
    print(f"Elapsed Training time: {elapsed_training_time}")
    
    mlflow.log_metric('training_time', elapsed_training_time)
    mlflow.log_metric('training_loss', average_loss)
    mlflow.log_metric('test_loss', average_test_loss)
    

    ############################################################################
    #########################  PLOT TRAINING RESULTS  ##########################
    ############################################################################

    x = np.linspace(1,num_epochs,num_epochs)
    plt.plot(x,train_hist,scalex=True, label="Training loss")
    plt.plot(x, test_hist, label="Test loss")
    plt.legend()
    plt.show(block=False)

    ############################################################################
    #################### PREDICT // FORECAST RESULTS  ##########################
    ############################################################################

    num_forecast_steps = 30
    sequence_to_plot = X_test.squeeze().cpu().numpy()
    print(sequence_to_plot.shape)
    historical_data = sequence_to_plot[-1]
    print(historical_data.shape)

    forecasted_values = []
    with torch.no_grad():
        for _ in range(num_forecast_steps):
            historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, n_tickers).float().to(device)
            # print(historical_data_tensor.shape)
            predicted_value = model(historical_data_tensor).cpu().numpy()[0, 0]
            # print(predicted_value.shape)
            forecasted_values.append(predicted_value)
            historical_data = np.roll(historical_data, shift=-1)
            historical_data[-1] = predicted_value

    last_date = test_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=30)

    ############################################################################
    ####################### PLOT PREDICT // FORECAST ###########################
    ############################################################################

    from pylab import rcParams

    plt.rcParams['figure.figsize'] = [14, 4]
    plt.plot(test_data.index[-100:], test_data[-100:], label="test_data", color="b")
    plt.plot(test_data.index[-30:], test_data[-30:], label='actual values', color='green')
    plt.plot(test_data.index[-1:].append(future_dates), np.concatenate([test_data[-1:], scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1)).flatten()]), label='forecasted values', color='red')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Time Series Forecasting')
    plt.grid(True)
    plt.show(block=False)

    ############################################################################
    #######################   PERFORMANCE METRICS   ###########################
    ############################################################################

    from sklearn.metrics import root_mean_squared_error, r2_score

    # Evaluate the model and calculate RMSE and R² score
    model.eval()
    with torch.no_grad():
        test_predictions = []
        for batch_X_test in X_test:
            batch_X_test = batch_X_test.to(device).unsqueeze(0)  # Add batch dimension
            test_predictions.append(model(batch_X_test).cpu().numpy().flatten()[0])

    test_predictions = np.array(test_predictions)

    # Calculate RMSE and R² score
    rmse = root_mean_squared_error(y_test.cpu().numpy(), test_predictions)
    r2 = r2_score(y_test.cpu().numpy(), test_predictions)

    print(f'RMSE: {rmse:.4f}')
    print(f'R² Score: {r2:.4f}')

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2-Score", r2)

    mlflow.end_run()
