# dlinear-multivariate-time-series
This repository implements a DLinear model for multivariate time series forecasting using PyTorch. The model efficiently captures long-term dependencies and trends while handling multivariate data with minimal computational overhead.

## Features

1. Implements the DLinear model for time series forecasting

2. Supports multivariate datasets (e.g., exchange rates, stock prices, weather data)

3. PyTorch-based, leveraging efficient deep learning techniques

4. Data preprocessing including scaling and sequence preparation

5. Model evaluation with MSE, MAE, and visualization of predictions

## Dataset

This implementation uses an exchange rate dataset, but it can be extended to other time series data. Ensure your dataset has:

- A timestamp column

- Multiple numerical feature columns

- Data in CSV format

## Quick Start

Install Dependencies
 pip install numpy pandas torch scikit-learn matplotlib
Clone the Repository
 git clone https://github.com/your-username/your-repo.git
 cd your-repo
Run the Notebook
Use Jupyter Notebook or execute the script:
 jupyter notebook dlinear_multivariate_time_series.ipynb

## Understanding DLinear

__What is DLinear?__

DLinear (Decomposition-Linear Model) is a simple yet effective deep learning model designed for time series forecasting. Unlike traditional RNN-based models (LSTMs, GRUs, or Transformers), DLinear utilizes a decomposition-based approach to separate the trend and seasonal components of time series data. By applying independent linear models to these components, DLinear achieves efficient and accurate forecasting with reduced computational cost.

__DLinear Architecture__

The model is based on two main components:

1. Decomposition Layer: Splits the input into trend and seasonal components.

2. Linear Layers: Each component is processed by a separate linear model.

Reconstruction Layer: Combines the trend and seasonal outputs to generate the final forecast.

This approach allows the model to focus on learning long-term patterns efficiently while reducing unnecessary complexity.

## Code Breakdown

1. Data Loading and Preprocessing

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

Loads the dataset from a CSV file.

Parses the date column and sets it as the index.

Uses StandardScaler to normalize the dataset before training.

2. Data Preparation

def prepare_data(df, seq_len=96, pred_len=14):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(len(df_scaled) - seq_len - pred_len):
        X.append(df_scaled[i:i+seq_len])
        y.append(df_scaled[i+seq_len:i+seq_len+pred_len])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

Converts the dataset into sequences of seq_len historical data points.

Defines prediction length (pred_len).

Uses torch tensors for training in PyTorch.

3. DLinear Model Implementation

class DLinear(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len):
        super(DLinear, self).__init__()
        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.seasonal_linear = nn.Linear(seq_len, pred_len)
    def forward(self, x):
        trend = self.trend_linear(x.mean(dim=-1))  # Extracts trend component
        seasonal = self.seasonal_linear(x - x.mean(dim=-1, keepdim=True))  # Extracts seasonal component
        return trend + seasonal  # Combines both components

Trend Component: Captures overall patterns using a linear layer.

Seasonal Component: Captures variations by subtracting the mean.

Final Output: Combines both components for the final prediction.

4. Training the Model

def train_model(model, dataloader, epochs=50, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

Uses Adam optimizer and MSE loss function.

Loops over mini-batches for training.

Optimizes model parameters using backpropagation.

5. Model Evaluation

def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        predictions, actuals = [], []
        for batch_x, batch_y in dataloader:
            predictions.append(model(batch_x))
            actuals.append(batch_y)
    return predictions, actuals

Evaluates the trained model on test data.

Uses torch.no_grad() for inference.

Compares predictions with ground truth.

## Results

The model achieves high accuracy on multivariate datasets with significantly reduced parameters compared to RNNs.
Sample Prediction

## Contributing

Pull requests are welcome! If you'd like to contribute:
Fork the repo
Create a new branch (git checkout -b feature-branch)
Commit your changes (git commit -m 'Add new feature')
Push to the branch (git push origin feature-branch)
Open a Pull Request

📜 License

This project is licensed under the MIT License.
