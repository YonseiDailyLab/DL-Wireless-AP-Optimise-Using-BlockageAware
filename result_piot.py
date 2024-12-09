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
from matplotlib import patches as patches

from model import Net
from datasets import CubeObstacle, CylinderObstacle, SvlDataset
from utils.tools import calc_sig_strength_gpu, calc_loss

logging.basicConfig(level=logging.INFO)

random_seed = 42
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
svl_model_path = "model/supervised/8500_model.pt"
usvl_model_path = "model/unsupervised/8500_model.pt"
result_path = "results/supervised"

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

    x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.25, random_state=42)

    logging.info(f"train shape: {x_train.shape}, val shape: {x_val.shape}")

    val_dataset = SvlDataset(x_val, y_val, dtype=torch.float32).to(device)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    svl_model = Net(val_dataset.x.shape[1], 1024, 4).to(device)
    usvl_model = Net(val_dataset.x.shape[1], 1024, 4, output_N=2).to(device)

    fig, ax = plt.subplots(5, 3, figsize=(20, 25), constrained_layout=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    with torch.no_grad():
        for i in range(5):
            svl_checkpoint = torch.load(svl_model_path)
            usvl_checkpoint = torch.load(usvl_model_path)
            svl_model.load_state_dict(svl_checkpoint['model_state_dict'])
            usvl_model.load_state_dict(usvl_checkpoint['model_state_dict'])
            
            svl_model.eval()
            x_val_batch = val_loader.dataset.x[i].unsqueeze(0)
            svl_pred = svl_model(x_val_batch)
            x_val_batch_reshaped = torch.from_numpy(scaler_x.inverse_transform(x_val_batch.cpu().numpy())).to(device, dtype=torch.float32).view(-1, 4, 3)
            svl_pred = torch.from_numpy(scaler_y.inverse_transform(svl_pred.cpu().numpy())).to(device, dtype=torch.float32)
            
            usvl_model.eval()
            usvl_pred = usvl_model(x_val_batch)
            x_val_batch_reshaped = torch.from_numpy(scaler_x.inverse_transform(x_val_batch.cpu().numpy())).to(device, dtype=torch.float32).view(-1, 4, 3)
            usvl_pred = torch.hstack((usvl_pred, torch.ones(usvl_pred.shape[0], 1).to(device) * 0.85)) * 200 - 100

            x_val_origin = scaler_x.inverse_transform(x_val_batch.cpu())
                
            val_gnd = torch.from_numpy(x_val_origin).to(device, dtype=torch.float32).view(4, 3)
            val_station = torch.tensor(y_val[0], dtype=torch.float32).unsqueeze(0).to(device)
            svl_station = svl_pred.clone().detach().to(device, dtype=torch.float32)
            usvl_station = usvl_pred.clone().detach().to(device, dtype=torch.float32)
                
            val_se = calc_sig_strength_gpu(val_station, val_gnd, obst_points).cpu().numpy()
            svl_pred_se = calc_sig_strength_gpu(svl_station,  val_gnd, obst_points).cpu().numpy()
            usvl_pred_se = calc_sig_strength_gpu(usvl_station,  val_gnd, obst_points).cpu().numpy()
            
            val_gnd = val_gnd.cpu().numpy()
            svl_station = svl_station.cpu().numpy()
            usvl_station = usvl_station.cpu().numpy()
            
            ax[i, 0].grid(True)
            ax[i, 0].set_xlim(-100, 100)
            ax[i, 0].set_ylim(-100, 100)
            ax[i, 0].set_aspect('equal')
            ax[i, 0].add_patch(patches.Rectangle((-30, 15), 60, 20, fill=True, color=colors[0]))
            ax[i, 0].add_patch(patches.Rectangle((-30, -25), 10, 35, fill=True, color=colors[1]))
            ax[i, 0].add_patch(patches.Circle((0, -30), 10, fill=True, color=colors[2]))
            
            for j in range(4):
                ax[i, 0].add_line(plt.Line2D([val_gnd[j, 0], svl_station[0, 0]], [val_gnd[j, 1], svl_station[0, 1]], color='b'))
                ax[i, 0].add_line(plt.Line2D([val_gnd[j, 0], usvl_station[0, 0]], [val_gnd[j, 1], usvl_station[0, 1]], color='g'))
            ax[i, 0].scatter(svl_station[0, 0], svl_station[0, 1], c='b', label='supervised learning')
            ax[i, 0].scatter(val_gnd[:, 0], val_gnd[:, 1], c='r', label='ground node')
            ax[i, 0].scatter(usvl_station[0, 0], usvl_station[0, 1], c='g', label='unsupervised learning')
            ax[i, 0].set_title(f"X-Y Plane Ground Node Set {i + 1} \n SVL SE: {svl_pred_se[0]:.2f} dB, USVL SE: {usvl_pred_se[0]:.2f} dB")
            ax[i, 0].set_xlabel('x')
            ax[i, 0].set_ylabel('y')
            
            ax[i, 0].legend(loc='upper right', fontsize=8)
            fig.tight_layout()
            
            ax[i, 1].grid(True)
            ax[i, 1].set_xlim(-100, 100)
            ax[i, 1].set_ylim(0, 100)
            ax[i, 1].set_aspect('equal')
            ax[i, 1].add_patch(patches.Rectangle((-30, 0), 60, 35, fill=True, color=colors[0]))
            ax[i, 1].add_patch(patches.Rectangle((-30, 0), 10, 45, fill=True, color=colors[1]))
            ax[i, 1].add_patch(patches.Rectangle((-10, 0), 20, 70, fill=True, color=colors[2]))
            for j in range(4):
                ax[i, 1].add_line(plt.Line2D([val_gnd[j, 0], svl_station[0, 0]], [val_gnd[j, 2], svl_station[0, 2]], color='b'))
                ax[i, 1].add_line(plt.Line2D([val_gnd[j, 0], usvl_station[0, 0]], [val_gnd[j, 2], usvl_station[0, 2]], color='g'))
            ax[i, 1].scatter(svl_station[0, 0], svl_station[0, 2], c='b', label='supervised learning')
            ax[i, 1].scatter(val_gnd[:, 0], val_gnd[:, 2], c='r', label='ground node')
            ax[i, 1].scatter(usvl_station[0, 0], usvl_station[0, 2], c='g', label='unsupervised learning')
            ax[i, 1].set_title(f"X-Z Plane Ground Node Set {i + 1} \n SVL SE: {svl_pred_se[0]:.2f} dB, USVL SE: {usvl_pred_se[0]:.2f} dB")
            ax[i, 1].set_xlabel('x')
            ax[i, 1].set_ylabel('z')
            ax[i, 1].legend(loc='upper right', fontsize=8)
            fig.tight_layout()
            
            ax[i, 2].grid(True)
            ax[i, 2].set_xlim(-100, 100)
            ax[i, 2].set_ylim(0, 100)
            ax[i, 2].set_aspect('equal')
            ax[i, 2].add_patch(patches.Rectangle((15, 0), 20, 35, fill=True, color=colors[0]))
            ax[i, 2].add_patch(patches.Rectangle((-25, 0), 35, 45, fill=True, color=colors[1]))
            ax[i, 2].add_patch(patches.Rectangle((-40, 0), 20, 70, fill=True, color=colors[2]))
            for j in range(4):
                ax[i, 2].add_line(plt.Line2D([val_gnd[j, 1], svl_station[0, 1]], [val_gnd[j, 2], svl_station[0, 2]], color='b'))
                ax[i, 2].add_line(plt.Line2D([val_gnd[j, 1], usvl_station[0, 1]], [val_gnd[j, 2], usvl_station[0, 2]], color='g'))
            ax[i, 2].scatter(svl_station[0, 1], svl_station[0, 2], c='b', label='supervised learning')
            ax[i, 2].scatter(val_gnd[:, 1], val_gnd[:, 2], c='r', label='ground node')
            ax[i, 2].scatter(usvl_station[0, 1], usvl_station[0, 2], c='g', label='unsupervised learning')
            ax[i, 2].set_title(f"Y-Z Plane Ground Node Set {i + 1} \n SVL SE: {svl_pred_se[0]:.2f} dB, USVL SE: {usvl_pred_se[0]:.2f} dB")
            ax[i, 2].set_xlabel('y')
            ax[i, 2].set_ylabel('z')
            ax[i, 2].legend(loc='upper right', fontsize=8)
            fig.tight_layout()
            
        plt.savefig("results/results1.png", dpi=300)
        plt.show()
        
        
        
        