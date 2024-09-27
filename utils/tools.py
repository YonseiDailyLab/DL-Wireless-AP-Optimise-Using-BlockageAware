import numpy as np

from utils.config import Hyperparameters as hparams
from datasets import Obstacle


def calc_dist(p1: np.ndarray, p2: np.ndarray, q: np.ndarray):
    p1p2 = p2 - p1
    distances = []
    for point in q.T:
        p1q = point - p1
        t = np.dot(p1q, p1p2) / np.dot(p1p2, p1p2)
        t = max(0, min(1, t))
        distances.append(np.linalg.norm((p1 + t * p1p2) - point))
    return np.array(distances)

def calc_sig_strength(station_pos: np.array, gn_pos: np.ndarray, obst: list[Obstacle]):
    num_gn = gn_pos.shape[0]
    sig = np.zeros(num_gn)

    for i in range(num_gn):
        dist = np.linalg.norm(station_pos - gn_pos[i])
        min_dist2obst = [np.min(calc_dist(station_pos, gn_pos[i], obst[j].points)) for j in range(len(obst))]

        bk_val = np.tanh(0.2 * np.min(min_dist2obst))
        chan_gain = bk_val * hparams.beta_1 / dist + (1 - bk_val) * hparams.beta_2 / (dist ** 1.65)
        sig[i] = hparams.P_AVG * chan_gain / hparams.noise

    return sum(sig)