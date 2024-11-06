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
from datasets import CubeObstacle, CylinderObstacle, SvlDataset
from utils.tools import calc_sig_strength_gpu

torch.random.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
writer = SummaryWriter()
tb_layout = {
    "SpectralEfficiency": {
        "SpectralEfficiency": ["Multiline", ["SpectralEfficiency/pred", "SpectralEfficiency/val", "SpectralEfficiency/rand", "SpectralEfficiency/zero"]]
    }
}
writer.add_custom_scalars(tb_layout)

batch_size = 1024*8

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

    model = Net(train_dataset.x.shape[1], 1024, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for epoch in range(10001):
        total_loss = []
        model.train()
        for x_batch, y_batch in tqdm(train_loader, desc=f"Train_Epoch {epoch}"):
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = torch.nn.functional.mse_loss(y_pred, y_batch)
            loss.backward()
            total_loss.append(loss.item())
            optimizer.step()
        # logging.info(f"Epoch: {epoch}, Loss: {np.mean(total_loss)}")
        writer.add_scalar("Loss/train", np.mean(total_loss), epoch)

        total_val_loss = []
        x_vals, y_val_preds, y_vals = [], [], []
        model.eval()
        with torch.no_grad():
            for x_val_batch, y_val_batch in tqdm(val_loader, desc=f"Val_Epoch {epoch}"):
                y_val_pred = model(x_val_batch)
                val_loss = torch.nn.functional.mse_loss(y_val_pred, y_val_batch)
                total_val_loss.append(val_loss.item())

                x_vals.append(x_val_batch.cpu())
                y_val_preds.append(y_val_pred.cpu())
                y_vals.append(y_val_batch.cpu())
        
        y_val_preds = torch.cat(y_val_preds).numpy()
        y_vals = torch.cat(y_vals).numpy()
        r2 = r2_score(y_vals, y_val_preds)
        # logging.info(f"Val Loss: {np.mean(total_val_loss)}, R2 Score: {r2}")
        writer.add_scalar("Loss/val", np.mean(total_val_loss), epoch)
        writer.add_scalar("R2/val", np.mean(r2), epoch)
        if epoch % 100 == 0:
            x_vals = torch.cat(x_vals).numpy()
            
            x_val_origin = scaler_x.inverse_transform(x_vals)
            y_val_origin = scaler_y.inverse_transform(y_vals)
            y_pred_origin = scaler_y.inverse_transform(y_val_preds)
            
            val_se_ls, pred_se_ls, rand_se_ls, zero_se_ls = [], [], [], []
            zero_station = torch.tensor([[0, 0, 70]]).to(device)
            
            for i in range(1000):
                val_gnd = torch.tensor(x_val_origin[i], dtype=torch.float32).reshape(4,3).to(device)
                val_station = torch.tensor(y_val_origin[i], dtype=torch.float32).unsqueeze(0).to(device)
                pred_station = torch.tensor(y_pred_origin[i], dtype=torch.float32).unsqueeze(0).to(device)
                rand_station = torch.tensor([[*torch.rand(2) * 200 - 100, 70]]).to(device)

                val_se_ls.append(calc_sig_strength_gpu(val_station, val_gnd, obst_points).cpu().numpy())
                pred_se_ls.append(calc_sig_strength_gpu(pred_station, val_gnd, obst_points).cpu().numpy())
                rand_se_ls.append(calc_sig_strength_gpu(rand_station, val_gnd, obst_points).cpu().numpy())
                zero_se_ls.append(calc_sig_strength_gpu(zero_station, val_gnd, obst_points).cpu().numpy())

            val_se_mean = np.mean(val_se_ls)
            pred_se_mean = np.mean(pred_se_ls)
            rand_se_mean = np.mean(rand_se_ls)
            zero_se_mean = np.mean(zero_se_ls)

            diff_val_pred = val_se_mean - pred_se_mean

            writer.add_scalar("SpectralEfficiency/pred", pred_se_mean, epoch)
            writer.add_scalar("SpectralEfficiency/val", val_se_mean, epoch)
            writer.add_scalar("SpectralEfficiency/rand", rand_se_mean, epoch)
            writer.add_scalar("SpectralEfficiency/zero", zero_se_mean, epoch)
            writer.add_scalar("SpectralEfficiency/diff", diff_val_pred, epoch)
                
            logging.info(f"val_sig: {np.mean(val_se_ls)}, pred_sig: {np.mean(pred_se_ls)}, val - pred: {np.mean(val_se_ls)-np.mean(pred_se_ls)}")


