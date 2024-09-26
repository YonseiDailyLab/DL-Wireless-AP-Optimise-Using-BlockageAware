import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from datasets import CubeObstacle, CylinderObstacle

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

    fig.tight_layout()
    plt.show()
