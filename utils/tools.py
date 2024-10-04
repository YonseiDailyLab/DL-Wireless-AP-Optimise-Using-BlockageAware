import numpy as np
import torch
from torch import Tensor

from utils.config import Hyperparameters as hparams
from datasets import Obstacle

def calc_dist(p1: np.ndarray, p2: np.ndarray, q: np.ndarray):
    p1p2 = p2 - p1
    p1q = q.T - p1
    t = np.clip(np.einsum('xy,y->x', p1q, p1p2) / np.dot(p1p2, p1p2), 0, 1)
    distances = np.linalg.norm((p1 + t[:, np.newaxis] * p1p2) - q.T, axis=1)
    return distances


def calc_sig_strength(station_pos: np.array, gn_pos: np.ndarray, obst: list[Obstacle]):
    num_gn = gn_pos.shape[0]
    sig = np.zeros(num_gn)

    for i in range(num_gn):
        dist = np.linalg.norm(station_pos - gn_pos[i])

        # Vectorized calculation for minimum distances to obstacles
        min_dist2obst = np.array([np.min(calc_dist(station_pos, gn_pos[i], obst[j].points)) for j in range(len(obst))])

        bk_val = np.tanh(0.2 * np.min(min_dist2obst))
        chan_gain = bk_val * hparams.beta_1 / dist + (1 - bk_val) * hparams.beta_2 / (dist ** 1.65)
        sig[i] = hparams.P_AVG * chan_gain / hparams.noise

    return np.sum(sig)/num_gn


def calc_dist_gpu(p1: Tensor, p2: Tensor, q: Tensor):
    p1p2 = p2 - p1
    p1q = q.T.unsqueeze(1) - p1
    dot_product = torch.sum(p1q * p1p2, dim=-1)  # [791, 40000]
    p1p2_norm_sq = torch.sum(p1p2 * p1p2, dim=-1)  # [40000]
    t = torch.clamp(dot_product / p1p2_norm_sq, 0, 1)
    closest_points = p1 + t.unsqueeze(-1) * p1p2
    distances = torch.linalg.norm(closest_points - q.T.unsqueeze(1), dim=-1)
    return distances


def calc_sig_strength_gpu(station_pos: Tensor, gn_pos: Tensor, obst: list[Obstacle]):
    num_gn = gn_pos.shape[0]
    sig = torch.zeros(num_gn)

    for i in range(num_gn):
        dist = torch.linalg.norm(station_pos - gn_pos[i])

        # Vectorized calculation for minimum distances to obstacles
        min_dist2obst = torch.tensor([torch.min(calc_dist_gpu(station_pos, gn_pos[i], obst[j].points)) for j in range(len(obst))])

        bk_val = torch.tanh(0.2 * torch.min(min_dist2obst))
        chan_gain = bk_val * hparams.beta_1 / dist + (1 - bk_val) * hparams.beta_2 / (dist ** 1.65)
        sig[i] = hparams.P_AVG * chan_gain / hparams.noise

    return torch.sum(sig)/num_gn