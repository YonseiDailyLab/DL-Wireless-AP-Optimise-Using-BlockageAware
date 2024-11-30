import os
import logging
import torch
import pandas as pd
import numpy as np
import wandb
from torch import layout
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from model import Net
from datasets import CubeObstacle, CylinderObstacle, SvlDataset
from utils.tools import calc_sig_strength_gpu

rand_seed = 42
torch.random.manual_seed(rand_seed)
np.random.seed(rand_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

batch_size = 1024*8
lr_ls = [5e-4, 1e-4, 5e-5, 1e-5, 2.5e-4]

wandb_project_name = "DL Based Wireless AP Optimise Using Blockage Aware Model"

logging.basicConfig(level=logging.INFO)
model_save_path = "model"

if __name__ == '__main__':
    obstacle_ls = [
        CubeObstacle(-30, 15, 35, 60, 20, 0.3),
        CubeObstacle(-30, -25, 45, 10, 35, 0.3),
        CylinderObstacle(0, -30, 70, 10, 0.3)
    ]
    obst_points = []
    for obstacle in obstacle_ls:
        obst_points.append(torch.tensor(obstacle.points, dtype=torch.float32))
    obst_points = torch.cat([op for op in obst_points], dim=1).mT.to(device)

    df = pd.concat([pd.read_csv('data/data1.csv'), pd.read_csv('data/data2.csv')])
    logging.info(df.shape)
    x = df.iloc[:, :12].values
    y = df.iloc[:, 12:].values

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y)

    logging.info(f"X shape: {x_scaled.shape}, Y shape: {y_scaled.shape}")
    logging.info(f"X: {x_scaled[0]}, Y: {y_scaled[0]}")

    x_train, x_val, y_train, y_val = train_test_split(x_scaled, y_scaled, test_size=0.25, random_state=42)

    logging.info(f"X_train shape: {x_train.shape}, Y_train shape: {y_train.shape}")
    logging.info(f"X_val shape: {x_val.shape}, Y_val shape: {y_val.shape}")

    train_dataset = SvlDataset(x_train, y_train, dtype=torch.float32).to(device)
    val_dataset = SvlDataset(x_val, y_val, dtype=torch.float32).to(device)
    
    print(train_dataset.x.shape, train_dataset.y.shape)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    for lr in lr_ls:
        
        wandb.init(project=wandb_project_name, name=f"supervised_test_lr:{lr}", config={
        "hidden_N": 1024,
        "hidden_L": 4,
        "batch_size": batch_size,
        "rand_seed": rand_seed,
        "lr": lr
        })
        logging.info(f"Learning rate: {lr}")
        model = Net(train_dataset.x.shape[1], 1024, 4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        for epoch in trange(3000):
            total_loss = []
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = torch.nn.functional.mse_loss(y_pred, y_batch)
                loss.backward()
                total_loss.append(loss.item())
                optimizer.step()


            wandb.log({"svl_test/Loss/train": np.mean(total_loss)})
        
        wandb.finish()
