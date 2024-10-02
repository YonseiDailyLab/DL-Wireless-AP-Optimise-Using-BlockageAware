import os
import logging
import pandas as pd
import numpy as np
from tqdm import trange

from datasets import CubeObstacle, CylinderObstacle, ChannelDataset
from utils.config import Hyperparameters as hparams
from utils.tools import calc_sig_strength

logging.basicConfig(level=logging.INFO)

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

if __name__ == "__main__":

    obstacle_ls = [
        CubeObstacle(-30, 15, 35, 60, 20),
        CubeObstacle(-30, -25, 45, 10, 35),
        CylinderObstacle(0, -30, 70, 10)
    ]
    gnd_nodes = []

    for _ in trange(200 * 200 * 4, desc="Generating gnd_nodes..."):
        temp = []
        while True:
            if len(temp) == hparams.num_node:
                break
            x = np.random.randint(-hparams.area_size // 2, hparams.area_size // 2)
            y = np.random.randint(-hparams.area_size // 2, hparams.area_size // 2)
            z = 0

            if (x, y) not in temp:
                is_inside = False
                for obstacle in obstacle_ls:
                    if is_inside := obstacle.is_inside(x, y, z):
                        break
                if not is_inside: temp.append((x, y, z))

        gnd_nodes.append(temp)

    result_ls = []
    X, Y = np.meshgrid(
        np.arange(-hparams.area_size//2, hparams.area_size//2),
        np.arange(-hparams.area_size//2, hparams.area_size//2),
        indexing='xy'
    )
    Z = np.full_like(X, 70)

    station_positions = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    gnd_nodes = np.array(gnd_nodes)
    for i in trange(200 * 200 * 4, desc="Generating dataset"):

        sig = np.array(
            [calc_sig_strength(station_pos, gnd_nodes[i], obstacle_ls) for station_pos in station_positions])
        sig = sig.reshape(hparams.area_size, hparams.area_size)

        max_idx = np.unravel_index(np.argmax(sig), sig.shape)
        # logging.info(f"Max Signal: {sig[max_idx]}, Index: {max_idx}")

        result_ls.append({
            "gnd1": gnd_nodes[0],
            "gnd2": gnd_nodes[1],
            "gnd3": gnd_nodes[2],
            "gnd4": gnd_nodes[3],
            "result": [int(max_idx[0]), int(max_idx[1]), 70],
            "sig": sig[max_idx]
        })


    createDirectory("data")
    df = pd.DataFrame(data=result_ls, columns=["gnd1", "gnd2", "gnd3", "gnd4", "result", "sig"])
    df.to_csv("data/data.csv", index=False)
    logging.info("Dataset created successfully!")