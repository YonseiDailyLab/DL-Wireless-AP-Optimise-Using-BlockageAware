import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CubeObstacle, CylinderObstacle, BlockageDataset
from utils.config import Hyperparameters as hparams
from utils.tools import calc_min_dist

logging.basicConfig(level=logging.WARNING)

def createDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_df(path: str, data: list):
    df = pd.DataFrame(data, columns=["gnd1_x","gnd1_y", "gnd1_z", "gnd2_x",
                                     "gnd2_y", "gnd2_z", "gnd3_x", "gnd3_y",
                                     "gnd3_z", "gnd4_x", "gnd4_y", "gnd4_z",
                                     "result_x", "result_y", "result_z",
                                     "sig1", "sig2", "sig3", "sig4", "sig_max"])
    createDirectory(path)
    df.to_csv(f"{path}/data.csv", index=False)

if __name__ == "__main__":
    result_ls = []

    # Create obstacles and convert to torch tensors
    obstacle_ls = [
        CubeObstacle(-30, 15, 35, 60, 20),
        CubeObstacle(-30, -25, 45, 10, 35),
        CylinderObstacle(0, -30, 70, 10)
    ]

    # Create dataset
    dataset = BlockageDataset(40000, obstacle_ls).to(hparams.device)
    logging.info(f"len(dataset): {len(dataset)}")
    dataloader = DataLoader(dataset)
    try:
        for i, data in enumerate(tqdm(dataloader)):
            station_pos, gnd_nodes, obst_points = data
            station_pos = station_pos.squeeze()
            gnd_nodes = gnd_nodes.squeeze()
            obst_points = obst_points.squeeze()

            min_dist = calc_min_dist(station_pos, gnd_nodes, obst_points)
            bk_val = torch.tanh(min_dist * 0.2)
            chan_gain = bk_val * hparams.beta_1 / min_dist + (1 - bk_val) * hparams.beta_2 / (min_dist ** 1.65)
            sig = hparams.P_AVG * chan_gain / hparams.noise
            sig_avg = torch.mean(sig, dim=1)
            logging.info(f"sig shape: {sig.shape}, sig_avg shape: {sig_avg.shape}")

            sig_max_idx = torch.argmax(sig_avg)
            sig_max = sig_avg[sig_max_idx]
            result = station_pos[sig_max_idx]

            np_gnd_nodes = gnd_nodes.cpu().numpy()
            np_result = result.cpu().numpy()
            np_sig = sig.cpu().numpy()
            np_sig_max = sig_max.cpu().numpy()

            result_ls.append([np_gnd_nodes[0, 0], np_gnd_nodes[0, 1], np_gnd_nodes[0, 2], np_gnd_nodes[1, 0],
                              np_gnd_nodes[1, 1], np_gnd_nodes[1, 2], np_gnd_nodes[2, 0], np_gnd_nodes[2, 1],
                              np_gnd_nodes[2, 2], np_gnd_nodes[3, 0], np_gnd_nodes[3, 1], np_gnd_nodes[3, 2],
                              np_result[0], np_result[1], np_result[2],
                              np_sig[sig_max_idx, 0], np_sig[sig_max_idx, 1],
                              np_sig[sig_max_idx, 2], np_sig[sig_max_idx, 3], np_sig_max])
            logging.info(f"Result: {result}, sig_max: {sig_max}")

    except KeyboardInterrupt:
        save_df("data", result_ls)

    save_df("data", result_ls)