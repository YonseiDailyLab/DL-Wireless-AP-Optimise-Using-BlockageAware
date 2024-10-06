import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from datasets import CubeObstacle, CylinderObstacle, BlockageDataset
from utils.config import Hyperparameters as hparams
from utils.tools import calc_min_dist

logging.basicConfig(level=logging.INFO)

def createDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":

    # Create obstacles and convert to torch tensors
    obstacle_ls = [
        CubeObstacle(-30, 15, 35, 60, 20),
        CubeObstacle(-30, -25, 45, 10, 35),
        CylinderObstacle(0, -30, 70, 10)
    ]

    # Create dataset
    dataset = BlockageDataset(40000, obstacle_ls)
    logging.info(f"len(dataset): {len(dataset)}")
    dataloader = DataLoader(dataset)
    for i, data in enumerate(dataloader):
        station_pos, gnd_nodes, obst_points = data
        station_pos = station_pos.squeeze()
        gnd_nodes = gnd_nodes.squeeze()
        obst_points = obst_points.squeeze()

        min_dist = calc_min_dist(station_pos, gnd_nodes, obst_points)
        logging.info(f"min_dist: {min_dist}")