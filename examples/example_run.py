import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from dlinear import DLinear, load_data, prepare_data, train_model, evaluate

# Example usage for exchange_rate.csv
if __name__ == "__main__":
    df = load_data("exchange_rate.csv")
    X, y, scaler = prepare_data(df)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DLinear(input_dim=X.shape[-1], seq_len=X.shape[1], pred_len=y.shape[1])
    train_model(model, dataloader, epochs=10)
    predictions, actuals = evaluate(model, dataloader)
    print("Predictions shape:", [p.shape for p in predictions])
    print("Actuals shape:", [a.shape for a in actuals])
