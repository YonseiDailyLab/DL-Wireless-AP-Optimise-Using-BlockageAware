import os
import logging
import torch
import pandas as pd
import numpy as np
from torch import layout
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from model import Net
from datasets import CubeObstacle, CylinderObstacle, UsvlDataset
from utils.tools import calc_sig_strength_gpu, calc_loss

logging.basicConfig(level=logging.INFO)

torch.random.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
batch_size = 1024

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
    # df = pd.read_csv('data/data.csv')
    logging.info(df.shape)
    x = df.iloc[:, :12].values

    scaler_x = MinMaxScaler(feature_range=(0, 1))

    x_scaled = scaler_x.fit_transform(x)

    x_train, x_val = train_test_split(x_scaled, test_size=0.25, random_state=42)

    logging.info(f"train shape: {x_train.shape}, val shape: {x_val.shape}")

    train_dataset = UsvlDataset(x_train, dtype=torch.float32).to(device)
    val_dataset = UsvlDataset(x_val, dtype=torch.float32).to(device)
    
    print(train_dataset.x.shape, val_dataset.x.shape)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Net(train_dataset.x.shape[1], 1024, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for epoch in range(10001):
        total_loss = []
        model.train()
        for x_batch in tqdm(train_loader, desc=f"Train_Epoch {epoch}"):
            optimizer.zero_grad()

            # 모델 예측 (배치 전체)
            y_pred = model(x_batch)

            # x_batch의 차원을 [batch_size, 4, 3] 형태로 변환하여 연산
            x_batch_reshaped = x_batch.view(-1, 4, 3)

            # calc_sig_strength_gpu 함수에서 배치 연산 수행
            loss = calc_loss(y_pred, x_batch_reshaped, obst_points)

            # 역전파 및 최적화
            loss.backward()
            total_loss.append(loss.item())
            optimizer.step()

        # Validation 루프
        total_val_loss = []
        x_vals, y_val_preds = [], []
        model.eval()
        with torch.no_grad():
            for x_val_batch in tqdm(val_loader, desc=f"Val_Epoch {epoch}"):
                y_val_pred = model(x_val_batch)
                x_val_batch_reshaped = x_val_batch.view(-1, 4, 3)
                val_loss = calc_loss(y_val_pred, x_val_batch_reshaped, obst_points)
                total_val_loss.append(val_loss.item())

                x_vals.append(x_val_batch.cpu())
                y_val_preds.append(y_val_pred.cpu())
            
        logging.info(f"Epoch: {epoch}, Loss: {np.mean(total_loss)}, Val Loss: {np.mean(total_val_loss)}")