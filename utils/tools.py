import numpy as np
import torch
from torch import Tensor

from utils.config import Hyperparameters as hparams
from datasets import Obstacle

def calc_dist(p1: np.ndarray, p2: np.ndarray, q: np.ndarray):
    v = p2 - p1
    w = q.T - p1
    t = np.clip(np.einsum('xy,y->x', w, v) / np.dot(v, v), 0, 1)
    distances = np.linalg.norm((p1 + t[:, np.newaxis] * v) - q.T, axis=1)
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
    v = p2 - p1
    w = q - p1
    t = torch.clamp(torch.inner(v, w) / torch.sum(v * v, dim = -1).unsqueeze(1), 0, 1)
    dist = torch.norm(p1.unsqueeze(0).unsqueeze(1) + t.unsqueeze(-1) * v.unsqueeze(1) - q, dim = -1)
    return dist

def calc_sig_strength_gpu(station_pos: Tensor, gn_pos: Tensor, obst: Tensor):

    xy_dist = torch.norm(station_pos - gn_pos, dim=-1)
    dist = calc_dist_gpu(station_pos, gn_pos, obst)
    bk_val = torch.tanh_(0.2 * torch.min(dist, dim=1).values)
    chan_gain = bk_val * hparams.beta_1 / xy_dist + (1 - bk_val) * hparams.beta_2 / (xy_dist ** 1.65)
    sig = hparams.P_AVG * chan_gain / hparams.noise
    return torch.mean(sig)

def calc_min_dist(station_pos: Tensor, gnd_nodes: Tensor, obst_points: Tensor):

    station_exp = station_pos.unsqueeze(1).unsqueeze(2)
    gnd_exp = gnd_nodes.unsqueeze(0).unsqueeze(2)
    obst_exp = obst_points.unsqueeze(0).unsqueeze(1)

    vec_station_gnd = gnd_exp - station_exp
    vec_station_obst = obst_exp - station_exp

    vec_station_gnd = vec_station_gnd.expand(-1, -1, obst_points.shape[0], -1)

    dot_product = torch.sum(vec_station_obst * vec_station_gnd, dim=-1)
    norm_sq = torch.sum(vec_station_gnd * vec_station_gnd, dim=-1)

    t = torch.clamp(dot_product / norm_sq, 0, 1)

    closest_p = station_exp + t.unsqueeze(-1) * vec_station_gnd
    dist = torch.linalg.norm(closest_p - obst_exp, dim=-1)
    min_dist = torch.min(dist, dim=-1).values

    return min_dist