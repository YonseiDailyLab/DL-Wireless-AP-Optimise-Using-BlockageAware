import os
import logging
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
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
model_path = "model/unsupervised"
result_path = "results/unsupervised"


def plot_results(ax, gnd, pred, val, obstacle_ls):
    for obstacle in obstacle_ls:
        obstacle.plot(ax)
    ax.scatter(gnd[0], gnd[1], gnd[2], c='r', marker='o', label='Ground Nodes')
    ax.scatter(gnd[3], gnd[4], gnd[5], c='r', marker='o')
    ax.scatter(gnd[6], gnd[7], gnd[8], c='r', marker='o')
    ax.scatter(gnd[9], gnd[10], gnd[11], c='r', marker='o')
    ax.scatter(pred[0], pred[1], pred[2], c='b', marker='o', label='Predicted (Model)')
    ax.scatter(val[0], val[1], val[2], c='g', marker='o', label='Bruteforce (Optimal)')
    ax.scatter(0, 0, 70, c='y', marker='o', label='Zero Point (0, 0, 70)')
    
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

    val_dataset = SvlDataset(x_val, y_val, dtype=torch.float32).to(device)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model_files = [f for f in os.listdir(model_path) if f.endswith(".pt")]

    model = Net(val_dataset.x.shape[1], 1024, 4, output_N=2).to(device)

    with torch.no_grad():
        for model_file in tqdm(model_files):
            checkpoint = torch.load(os.path.join(model_path, model_file))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            x_val_batch = val_loader.dataset.x[0].unsqueeze(0)
            y_val_pred = model(x_val_batch)
            x_val_batch_reshaped = torch.from_numpy(scaler_x.inverse_transform(x_val_batch.cpu().numpy())).to(device, dtype=torch.float32).view(-1, 4, 3)
            y_val_pred = torch.hstack((y_val_pred, torch.ones(y_val_pred.shape[0], 1).to(device) * 0.85)) * 200 - 100

            x_val_origin = scaler_x.inverse_transform(x_val_batch.cpu())
            
            val_gnd = torch.from_numpy(x_val_origin).to(device, dtype=torch.float32).view(4, 3)
            val_station = torch.tensor(y_val[0], dtype=torch.float32).unsqueeze(0).to(device)
            pred_station = y_val_pred.clone().detach().to(device, dtype=torch.float32)
            
            val_se = calc_sig_strength_gpu(val_station, val_gnd, obst_points).cpu().numpy()
            pred_se = calc_sig_strength_gpu(pred_station, val_gnd, obst_points).cpu().numpy()
            
            epoch = checkpoint['epoch']

            # 원래의 뷰
            fig = plt.figure()
            fig.suptitle(f"Epoch {epoch} Results \n Pred SE: {pred_se[0]:.2f}dB, Val SE: {val_se[0]:.2f}dB \n \n ", fontsize=10)
            
            ax = fig.add_subplot(111, projection='3d')
            plot_results(ax, x_val_origin[0], y_val_pred[0].cpu().numpy(), y_val[0], obstacle_ls)
            ax.legend(
                loc='upper left', 
                fontsize=6  # Smaller font size
            )
            fig.tight_layout()
            plt.savefig(f"{result_path}/{epoch}_original_view.png", dpi=150)
            plt.savefig(f"{result_path}/{epoch}_original_view.eps", dpi=150)
            plt.savefig(f"{result_path}/{epoch}_original_view.svg", dpi=150)

            fig, axs = plt.subplots(2, 3, subplot_kw={'projection': '3d'}, figsize=(18, 12))
            fig.suptitle(f"Epoch {epoch} Results \n Pred SE: {pred_se[0]:.2f}dB, Val SE: {val_se[0]:.2f}dB \n \n", fontsize=14)
            
            # 첫 번째 행: XY, XZ, YZ 뷰
            views = [('XY', (90, -90, 0)),
                    ('XZ', (0, -90, 0)),
                    ('YZ', (0, 0, 0))]

            for i, (plane, angles) in enumerate(views):
                ax = axs[0, i]
                ax.view_init(elev=angles[0], azim=angles[1])
                plot_results(ax, x_val_origin[0], y_val_pred[0].cpu().numpy(), y_val[0], obstacle_ls)
                ax.set_title(f'{plane} View')

            # 두 번째 행: -XY, -XZ, -YZ 뷰
            views = [('-XY', (-90, 90, 0)),
                    ('-XZ', (0, 90, 0)),
                    ('-YZ', (0, 180, 0))]

            for i, (plane, angles) in enumerate(views):
                ax = axs[1, i]
                ax.view_init(elev=angles[0], azim=angles[1])
                plot_results(ax, x_val_origin[0], y_val_pred[0].cpu().numpy(), y_val[0], obstacle_ls)
                ax.set_title(f'{plane} View')

            axs[0,0].legend(
                loc='upper left', 
                fontsize=10  # Smaller font size
            )
            
            # 전체 레이아웃 조정 및 저장
            fig.tight_layout()
            plt.savefig(f"{result_path}/{epoch}_primary_views.png", dpi=150)
            plt.savefig(f"{result_path}/{epoch}_primary_views.eps", dpi=150)
            plt.savefig(f"{result_path}/{epoch}_primary_views.svg", dpi=150)
            plt.show()