# import numpy as np
import torch
from torch import Tensor

from utils.config import Hyperparameters as hparams
from datasets import Obstacle

def calc_dist(p1: Tensor, p2: Tensor, q: Tensor):
    p1 = p1.float().to(hparams.device)
    p2 = p2.float().to(hparams.device)
    q = q.float().to(hparams.device)
    p1p2 = p2 - p1
    distances = []
    for point in q.T:
        p1q = point - p1
        t = torch.dot(p1q, p1p2) / torch.dot(p1p2, p1p2)
        t = max(0, min(1, t))
        distances.append(torch.linalg.norm((p1 + t * p1p2) - point))
    return torch.tensor(distances).to(hparams.device)

def calc_sig_strength(station_pos: Tensor, gn_pos: Tensor, obst: list[Obstacle]):
    station_pos = station_pos.float().to(hparams.device)
    gn_pos = gn_pos.float().to(hparams.device)
    num_gn = gn_pos.shape[0]
    sig = torch.zeros(num_gn).to(hparams.device)

    for i in range(num_gn):
        dist = torch.linalg.norm(station_pos - gn_pos[i])
        min_dist2obst = torch.tensor([torch.min(calc_dist(station_pos, gn_pos[i], torch.tensor(obst[j].points).float().to(hparams.device))) for j in range(len(obst))]).to(hparams.device)

        bk_val = torch.tanh(0.2 * torch.min(min_dist2obst))
        chan_gain = bk_val * hparams.beta_1 / dist + (1 - bk_val) * hparams.beta_2 / (dist ** 1.65)
        sig[i] = hparams.P_AVG * chan_gain / hparams.noise

    return sum(sig)