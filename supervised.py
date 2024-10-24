import os
import logging
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from model import Net

torch.random.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)

batch_size = 1024

class SvlDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

if __name__ == '__main__':

    df = pd.read_csv('data/data.csv')
    x = df.iloc[:, :12].values
    y = df.iloc[:, 12:].values

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y)

    logging.info(f"X shape: {x_scaled.shape}, Y shape: {y_scaled.shape}")
    logging.info(f"X: {x_scaled[0]}, Y: {y_scaled[0]}")

    x_train, x_val, y_train, y_val = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

    logging.info(f"X_train shape: {x_train.shape}, Y_train shape: {y_train.shape}")
    logging.info(f"X_val shape: {x_val.shape}, Y_val shape: {y_val.shape}")

    train_dataset = SvlDataset(x_train, y_train)
    val_dataset = SvlDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Net(4, 1024, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for epoch in range(100000):
        total_loss = []
        model.train()
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = torch.nn.functional.mse_loss(y_pred, y_batch)
            loss.backward()
            total_loss.append(loss.item())
            optimizer.step()
        logging.info(f"Epoch: {epoch}, Loss: {np.mean(total_loss)}")

        total_val_loss = []
        total_r2 = []
        model.eval()
        with torch.no_grad():
            for x_val_batch, y_val_batch in val_loader:
                y_val_pred = model(x_val_batch)
                val_loss = torch.nn.functional.mse_loss(y_val_pred, y_val_batch)
                r2 = r2_score(y_val_batch.cpu().numpy(), y_val_pred.cpu().numpy())
                total_val_loss.append(val_loss.item())
                total_r2.append(r2)
            logging.info(f"Val Loss: {np.mean(total_val_loss)}, R2 Score: {np.mean(total_r2)}")
