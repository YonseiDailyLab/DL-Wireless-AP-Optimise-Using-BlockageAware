import logging
import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CubeObstacle, CylinderObstacle, ChannelDataset
from utils.config import Hyperparameters as hparams

np.random.seed(42)
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # Create dataset
    n = 1
    obstacle_ls = [
        CubeObstacle(-30, 15, 35, 60, 20),
        CubeObstacle(-30, -25, 45, 10, 35),
        CylinderObstacle(0, -30, 70, 10)
    ]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for obstacle in obstacle_ls:
        obstacle.plot(ax)
        logging.info(f"Obstacle.points.shape: {obstacle.points.shape}")

    fig.tight_layout()
    plt.show()

    points = np.concatenate((obstacle_ls[0].points, obstacle_ls[1].points, obstacle_ls[2].points), axis=1)
    x = torch.tensor(points, dtype=torch.float32).to(hparams.device)
    logging.info(f"x: {x.shape}")

    # Create dataset
    train_dataset = ChannelDataset(hparams.num_samples, hparams.num_node, hparams.area_size, hparams.v_speed)
    logging.info(f"len(dataset): {len(train_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch, shuffle=True)

    # Brute Force coord

    X, Y = np.meshgrid(
        np.arange(-hparams.area_size // 2, hparams.area_size // 2),
        np.arange(-hparams.area_size // 2, hparams.area_size // 2),
        indexing='xy'
    )
    Z = np.full_like(X, 70)

    station_positions = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)