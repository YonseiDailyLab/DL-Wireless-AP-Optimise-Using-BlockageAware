import logging
from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt

class DotCloud:
    def __init__(self, x: int, y: int, height: int):
        self.shape = {"x": x,
                      "y": y,
                      "z": height}

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
    def show2plt(self):
        pass




class CubeCloud(DotCloud):
    def __init__(self, x: int, y: int, height: int, width: int, depth: int, n: int=20):
        super().__init__(x, y, height)
        self.shape["width"] = width
        self.shape["height"] = depth

        __points = [
                    # front face
                    np.array([[x + width * np.random.rand() for _ in range(n)],
                              [y] * np.ones(n, ),
                              [height * np.random.rand() for _ in range(n)]]),
                    # back face
                    np.array([[x + width * np.random.rand() for _ in range(n)],
                              [(y + depth)] * np.ones(n, ),
                              [height * np.random.rand() for _ in range(n)]]),
                    # left face
                    np.array([[x] * np.ones(n, ),
                              [y + depth * np.random.rand() for _ in range(n)],
                              [height * np.random.rand() for _ in range(n)]]),
                    # right face
                    np.array([[x + width] * np.ones(n, ),
                              [y + depth * np.random.rand() for _ in range(n)],
                              [height * np.random.rand() for _ in range(n)]]),
                    # top face
                    np.array([[x + width * np.random.rand() for _ in range(n)],
                              [y + depth * np.random.rand() for _ in range(n)],
                              [height] * np.ones(n, )])
                    ]
        # concatenate all
        self.points = np.concatenate(__points)

    @property
    def width(self):
        return self.shape["width"]

    @property
    def depth(self):
        return self.shape["depth"]

    def __str__(self):
        return f"CubeCloud: {self.shape}"

    def show2plt(self):
        pass


class CylinderCloud(DotCloud):
    def __init__(self, x: int, y: int, height: int, radius: int, n: int=20):
        super().__init__(x, y, height)
        self.shape["radius"] = radius

        __points = [
                    # top face
                    np.array([[x + radius * np.cos(2 * np.pi * np.random.rand()) for _ in range(n)],
                              [y + radius * np.sin(2 * np.pi * np.random.rand()) for _ in range(n)],
                              [height] * np.ones(n, )]),
                    # bottom face
                    np.array([[x + radius * np.cos(2 * np.pi * np.random.rand()) for _ in range(n)],
                              [y + radius * np.sin(2 * np.pi * np.random.rand()) for _ in range(n)],
                              [0] * np.ones(n, )]),
                    # side face
                    np.array([[x + radius * np.cos(2 * np.pi * np.random.rand()) for _ in range(n)],
                              [y + radius * np.sin(2 * np.pi * np.random.rand()) for _ in range(n)],
                              [height * np.random.rand() for _ in range(n)]])
                    ]
        # concatenate all
        self.points = np.concatenate(__points)

    @property
    def radius(self):
        return self.shape["radius"]

    def __str__(self):
        return f"CylinderCloud: {self.shape}"

    def show2plt(self):
        pass


if __name__ == "__main__":
    cube = CubeCloud(0, 0, 0, 10, 10)
    print(cube.shape)
    print(cube.points)
