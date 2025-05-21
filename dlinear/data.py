import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """Load a CSV file, parse dates, and set the date column as index."""
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

def prepare_data(df: pd.DataFrame, seq_len: int = 96, pred_len: int = 14) -> Tuple[torch.Tensor, torch.Tensor, StandardScaler]:
    """Scale data and create input/output sequences for time series modeling."""
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(len(df_scaled) - seq_len - pred_len):
        X.append(df_scaled[i:i+seq_len])
        y.append(df_scaled[i+seq_len:i+seq_len+pred_len])
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        scaler
    )
