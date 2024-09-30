import logging
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import torch

from datasets import CubeObstacle, CylinderObstacle, ChannelDataset
from utils.config import Hyperparameters as hparams
from utils.tools import calc_sig_strength

np.random.seed(52)
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

    gnd_nodes = []
    while True:
        if len(gnd_nodes) == hparams.num_node:
            break
        x = np.random.randint(-hparams.area_size // 2, hparams.area_size // 2)
        y = np.random.randint(-hparams.area_size // 2, hparams.area_size // 2)
        z = 0

        if (x, y) not in gnd_nodes:
            is_inside = False
            for obstacle in obstacle_ls:
                if is_inside := obstacle.is_inside(x, y, z):
                    break
            if not is_inside: gnd_nodes.append((x, y, z))

    gnd_nodes = np.array(gnd_nodes)
    logging.info(f"gnd_nodes: {gnd_nodes}")
    ax.scatter(gnd_nodes[:, 0], gnd_nodes[:, 1], gnd_nodes[:, 2], c='r')

    fig.tight_layout()
    plt.show()

    X, Y = np.meshgrid(
        np.arange(-hparams.area_size // 2, hparams.area_size // 2),
        np.arange(-hparams.area_size // 2, hparams.area_size // 2),
        indexing='xy'
    )
    Z = np.full_like(X, 70)

    station_positions = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    sig = np.array([calc_sig_strength(station_pos, gnd_nodes, obstacle_ls) for station_pos in tqdm(station_positions)])
    sig = sig.reshape(hparams.area_size, hparams.area_size)

    max_idx = np.unravel_index(np.argmax(sig), sig.shape)
    print(f"Max Signal: {sig[max_idx]}, Index: {max_idx}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for obstacle in obstacle_ls:
        obstacle.plot(ax)
        logging.info(f"Obstacle.points.shape: {obstacle.points.shape}")

    ax.scatter(gnd_nodes[:, 0], gnd_nodes[:, 1], gnd_nodes[:, 2], c='r')
    ax.scatter(max_idx[0]-(hparams.area_size//2), max_idx[1]-(hparams.area_size//2), 70, c='g')

    fig.tight_layout()
    plt.show()
