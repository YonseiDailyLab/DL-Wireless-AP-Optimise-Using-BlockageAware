import os
import logging
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
import wandb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from model import Net
from datasets import CubeObstacle, CylinderObstacle, SvlDataset
from utils.tools import calc_sig_strength_gpu, calc_loss

logging.basicConfig(level=logging.INFO)

random_seed = 42
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
batch_size = 2**13
model_save_path = "model/unsupervised"

wandb_project_name = "DL Based Wireless AP Optimise Using Blockage Aware Model"
wandb.init(project=wandb_project_name, config={
    "hidden_N": 1024,
    "hidden_L": 4,
    "batch_size": batch_size,
    "rand_seed": random_seed
})

def plot_results(ax, gnd, pred, val, obstacle_ls):
    for obstacle in obstacle_ls:
        obstacle.plot(ax)
    ax.scatter(gnd[0], gnd[1], gnd[2], c='r', marker='o')
    ax.scatter(gnd[3], gnd[4], gnd[5], c='r', marker='o')
    ax.scatter(gnd[6], gnd[7], gnd[8], c='r', marker='o')
    ax.scatter(gnd[9], gnd[10], gnd[11], c='r', marker='o')
    ax.scatter(pred[0], pred[1], pred[2], c='b', marker='o')
    ax.scatter(val[0], val[1], val[2], c='g', marker='o')
    ax.scatter(0, 0, 70, c='y', marker='o')
    
    ax.plot([gnd[0], pred[0]], [gnd[1], pred[1]], [gnd[2], pred[2]], c='b')
    ax.plot([gnd[3], pred[0]], [gnd[4], pred[1]], [gnd[5], pred[2]], c='b')
    ax.plot([gnd[6], pred[0]], [gnd[7], pred[1]], [gnd[8], pred[2]], c='b')
    ax.plot([gnd[9], pred[0]], [gnd[10], pred[1]], [gnd[11], pred[2]], c='b')
    
    ax.plot([gnd[0], val[0]], [gnd[1], val[1]], [gnd[2], val[2]], c='g')
    ax.plot([gnd[3], val[0]], [gnd[4], val[1]], [gnd[5], val[2]], c='g')
    ax.plot([gnd[6], val[0]], [gnd[7], val[1]], [gnd[8], val[2]], c='g')
    ax.plot([gnd[9], val[0]], [gnd[10], val[1]], [gnd[11], val[2]], c='g')

    return ax

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

    x_scaled = scaler_x.fit_transform(x)

    x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.25, random_state=42)

    logging.info(f"train shape: {x_train.shape}, val shape: {x_val.shape}")

    train_dataset = SvlDataset(x_train, y_train, dtype=torch.float32).to(device)
    val_dataset = SvlDataset(x_val, y_val, dtype=torch.float32).to(device)
    
    print(train_dataset.x.shape, val_dataset.x.shape)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Net(train_dataset.x.shape[1], 1024, 4, output_N=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for epoch in range(10001):
        total_loss = []
        model.train()
        for x_batch, _ in tqdm(train_loader, desc=f"Train_Epoch {epoch}"):
            optimizer.zero_grad()

            # 모델 예측 (배치 전체)
            y_pred = model(x_batch)

            # x_batch의 차원을 [batch_size, 4, 3] 형태로 변환하여 연산
            x_batch_reshaped = torch.tensor(scaler_x.inverse_transform(x_batch.cpu()), device=device, dtype=torch.float32).view(-1, 4, 3)
            y_pred = torch.hstack((y_pred, torch.ones(y_pred.shape[0], 1, device=device) * 0.85))*200-100
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
            for x_val_batch, _ in tqdm(val_loader, desc=f"Val_Epoch {epoch}"):
                y_val_pred = model(x_val_batch)
                x_val_batch_reshaped = torch.tensor(scaler_x.inverse_transform(x_val_batch.cpu()), device=device, dtype=torch.float32).view(-1, 4, 3)
                y_val_pred = torch.hstack((y_val_pred, torch.ones(y_val_pred.shape[0], 1).to(device) * 0.85)) * 200 - 100
                val_loss = calc_loss(y_val_pred, x_val_batch_reshaped, obst_points)
                total_val_loss.append(val_loss.item())

                x_vals.append(x_val_batch.cpu())
                y_val_preds.append(y_val_pred.cpu())
            
        logging.info(f"Epoch: {epoch}, Loss: {np.mean(total_loss)}, Val Loss: {np.mean(total_val_loss)}")

        if epoch % 100 == 0:
            x_vals = torch.cat(x_vals).numpy()
            y_val_preds = torch.cat(y_val_preds).numpy()

            x_val_origin = scaler_x.inverse_transform(x_vals)

            val_se_ls, pred_se_ls, rand_se_ls, zero_se_ls = [], [], [], []
            zero_station = torch.tensor([[0, 0, 70]]).to(device)

            for i in range(1000):

                val_gnd = torch.tensor(x_val_origin[i], dtype=torch.float32).reshape(4, 3).to(device)
                val_station = torch.tensor(y_val[i], dtype=torch.float32).unsqueeze(0).to(device)
                pred_station = torch.tensor(y_val_preds[i], dtype=torch.float32).unsqueeze(0).to(device)
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

            wandb.log({
                "SpectralEfficiency/pred": pred_se_mean,
                "SpectralEfficiency/val": val_se_mean,
                "SpectralEfficiency/rand": rand_se_mean,
                "SpectralEfficiency/zero": zero_se_mean,
                "SpectralEfficiency/diff": diff_val_pred,
                "Loss/train": np.mean(total_loss),
                "Loss/val": np.mean(total_val_loss)
            })
            logging.info(f"x_val_origin: {x_val_origin[0]}, y_val_preds: {y_val_preds[0]}, y_val: {y_val[0]}")
            logging.info(
                f"val_sig: {np.mean(val_se_ls)}, pred_sig: {np.mean(pred_se_ls)}, val - pred: {np.mean(val_se_ls) - np.mean(pred_se_ls)}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': np.mean(total_val_loss),
            }, os.path.join(model_save_path, f'{epoch}_model.pt'))
            logging.info(f"Training completed. {epoch} model saved.")

            # 원래의 뷰
            fig = plt.figure()
            fig.suptitle(f"Epoch {epoch} Results \n Pred SE: {pred_se_ls[0][0]:.2f}dB, Val SE: {val_se_ls[0][0]:.2f}dB", fontsize=8)
            
            ax = fig.add_subplot(111, projection='3d')
            plot_results(ax, x_val_origin[0], y_val_preds[0], y_val[0], obstacle_ls)
            fig.tight_layout()
            plt.savefig(f"results/unsupervised/{epoch}_original_view.png")

            fig, axs = plt.subplots(2, 3, subplot_kw={'projection': '3d'}, figsize=(18, 12))
            fig.suptitle(f"Epoch {epoch} Results \n Pred SE: {pred_se_ls[0][0]:.2f}dB, Val SE: {val_se_ls[0][0]:.2f}dB", fontsize=16)
            
            # 첫 번째 행: XY, XZ, YZ 뷰
            views = [('XY', (90, -90, 0)),
                    ('XZ', (0, -90, 0)),
                    ('YZ', (0, 0, 0))]

            for i, (plane, angles) in enumerate(views):
                ax = axs[0, i]
                ax.view_init(elev=angles[0], azim=angles[1], roll=angles[2])
                plot_results(ax, x_val_origin[0], y_val_preds[0], y_val[0], obstacle_ls)
                ax.set_title(f'{plane} View')

            # 두 번째 행: -XY, -XZ, -YZ 뷰
            views = [('-XY', (-90, 90, 0)),
                    ('-XZ', (0, 90, 0)),
                    ('-YZ', (0, 180, 0))]

            for i, (plane, angles) in enumerate(views):
                ax = axs[1, i]
                ax.view_init(elev=angles[0], azim=angles[1], roll=angles[2])
                plot_results(ax, x_val_origin[0], y_val_preds[0], y_val[0], obstacle_ls)
                ax.set_title(f'{plane} View')

            # 전체 레이아웃 조정 및 저장
            fig.tight_layout()
            plt.savefig(f"results/unsupervised/{epoch}_primary_views.png")
            plt.show()

        else:
            wandb.log({"Loss/train": np.mean(total_loss),
                       "Loss/val": np.mean(total_val_loss)})

    wandb.finish()
    