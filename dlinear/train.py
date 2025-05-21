import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from typing import Tuple

def train_model(model: nn.Module, dataloader: DataLoader, epochs: int = 50, lr: float = 0.001) -> None:
    """Train the DLinear model using MSE loss and Adam optimizer."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

def evaluate(model: nn.Module, dataloader: DataLoader) -> Tuple[list, list]:
    """Evaluate the model and return predictions and actuals."""
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            predictions.append(model(batch_x))
            actuals.append(batch_y)
    return predictions, actuals
