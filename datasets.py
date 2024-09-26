import logging
from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset

class Obstacle:
    def __init__(self, x: int, y: int, height: int):
        self.shape = {"x": x,
                      "y": y,
                      "height": height}

    @property
    def x(self):
        return self.shape["x"]

    @property
    def y(self):
        return self.shape["y"]

    @property
    def height(self):
        return self.shape["height"]

    def __str__(self):
        return f"DotCloud: {self.shape}"

    @abstractmethod
    def plot(self, ax):
        pass


class CubeObstacle(Obstacle):
    def __init__(self, x: int, y: int, height: int, width: int, depth: int):
        super().__init__(x, y, height)
        self.shape["width"] = width
        self.shape["depth"] = depth

        top = width*depth//4
        fb = width*height//4
        lr = depth*height//4

        __points = [
                    # front face
                    np.array([x + width * np.random.rand(fb),
                              [y] * np.ones(fb, ),
                              height * np.random.rand(fb)]),
                    # back face
                    np.array([x + width * np.random.rand(fb),
                              [(y + depth)] * np.ones(fb, ),
                              height * np.random.rand(fb)]),
                    # left face
                    np.array([[x] * np.ones(lr, ),
                              y + depth * np.random.rand(lr),
                              height * np.random.rand(lr)]),
                    # right face
                    np.array([[x + width] * np.ones(lr, ),
                              y + depth * np.random.rand(lr),
                              height * np.random.rand(lr)]),
                    # top face
                    np.array([x + width * np.random.rand(top),
                              y + depth * np.random.rand(top),
                              [height] * np.ones(top, )])
                    ]
        # concatenate all
        self.points = np.concatenate(__points, axis=1)

    @property
    def width(self):
        return self.shape["width"]

    @property
    def depth(self):
        return self.shape["depth"]

    def __str__(self):
        return f"CubeCloud: {self.shape}"

    def plot(self, ax):
        return ax.scatter(self.points[0], self.points[1], self.points[2])


class CylinderObstacle(Obstacle):
    def __init__(self, x: int, y: int, height: int, radius: int):
        super().__init__(x, y, height)
        self.shape["radius"] = radius

        t_num = int(radius**2*np.pi)//2
        s_num = int(2*radius*np.pi*height)//2

        r_top = radius * np.sqrt(np.random.rand(t_num))
        theta_top = np.random.rand(t_num) * 2 * np.pi
        angles_side = np.linspace(0, 2 * np.pi, s_num, endpoint=False)
        __points = [
            # top face
            np.array([x + r_top * np.cos(theta_top),
                      y + r_top * np.sin(theta_top),
                      [height] * t_num]),
            # side faces
            np.array([x + radius * np.cos(angles_side),
                      y + radius * np.sin(angles_side),
                      height * np.random.rand(s_num)])
        ]
        self.points = np.concatenate(__points, axis=1)

    @property
    def radius(self):
        return self.shape["radius"]

    def __str__(self):
        return f"CylinderCloud: {self.shape}"

    def plot(self, ax):
        return ax.scatter(self.points[0], self.points[1], self.points[2])


class ChannelDataset(Dataset):
    def __init__(self, x_data: np.ndarray, gnd_nodes: int, area_size: int, v_speed: int):
        super(ChannelDataset, self).__init__()
        self.x_data = torch.zeros(x_data, gnd_nodes + 1, 4)
        for i in range(x_data):
            temp_dist = area_size * (torch.rand(gnd_nodes + 1, 4) - 0.5)

            ## velocity of vehicle
            temp_x_vel = torch.rand(gnd_nodes + 1, 1)
            temp_vel_x = v_speed * torch.cos(2 * torch.pi * temp_x_vel)
            temp_vel_y = v_speed * torch.sin(2 * torch.pi * temp_x_vel)

            temp_dist[:, 2:3] = temp_vel_x
            temp_dist[:, 3:4] = temp_vel_y

            temp_dist[gnd_nodes, :] = 0

            self.x_data[i] = temp_dist

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x_data[idx])

if __name__ == "__main__":
    cube = CubeObstacle(0, 0, 0, 10, 10)
    print(cube.shape)
    print(cube.points)
