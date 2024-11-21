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
        snr = hparams.P_AVG * chan_gain / hparams.noise
        se = np.log2(1 + snr)
        sig[i] = se

    return np.mean(sig)

def calc_dist_gpu(p1: Tensor, p2: Tensor, q: Tensor):
    v = p2[None, :, :] - p1[:, None, :]
    w = q[None, :, :] - p1[:, None, :]
    v_norm_squared = (v ** 2).sum(dim=2, keepdim=True)
    dot_product = (v[:, :, None, :] * w[:, None, :, :]).sum(dim=3)
    t = torch.clamp(dot_product / v_norm_squared, 0, 1)
    p = p1[:, None, None, :] + t[..., None] * v[:, :, None, :]
    dist = torch.norm(p - q[None, None, :, :], dim=3)
    return dist

def calc_sig_strength_gpu(station_pos: Tensor, gn_pos: Tensor, obst: Tensor):
    dist = calc_dist_gpu(station_pos, gn_pos, obst)
    bk_val = torch.tanh(torch.min(dist, dim=-1).values*0.2)

    norm = torch.norm(station_pos.unsqueeze(1) - gn_pos.unsqueeze(0), dim=-1)
    chan_gain = bk_val * hparams.beta_1 / norm + (1 - bk_val) * hparams.beta_2 / (norm ** 1.65)

    snr = hparams.P_AVG * chan_gain / hparams.noise
    se = torch.log2(1 + snr) # Data rate, Spectral Efficiency
    
    return torch.mean(se, dim=1)

def calc_loss(y_pred: Tensor, x_batch: Tensor, obst_points: Tensor):
    x_batch_reshaped = x_batch.view(-1, 4, 3)
    
    p1, p2, q = y_pred, x_batch_reshaped, obst_points

    # v와 w의 차원 수정
    v = p2 - p1.unsqueeze(1)  # [batch_size, 4, 3]
    w = q.unsqueeze(0) - p1.unsqueeze(1)  # [batch_size, N_c, 3]

    v_norm_squared = (v ** 2).sum(dim=2, keepdim=True)  # [batch_size, 4, 1]
    dot_product = (v.unsqueeze(2) * w.unsqueeze(1)).sum(dim=3)  # [batch_size, 4, N_c]

    t = torch.clamp(dot_product / v_norm_squared, 0, 1)  # [batch_size, 4, N_c]

    p = p1.unsqueeze(1).unsqueeze(2) + t.unsqueeze(-1) * v.unsqueeze(2)  # [batch_size, 4, N_c, 3]

    dist = torch.norm(p - q.unsqueeze(0).unsqueeze(0), dim=3)  # [batch_size, 4, N_c]

    min_dist2obst = torch.min(dist, dim=2).values  # [batch_size, 4]
    bk_val = torch.tanh(0.2 * min_dist2obst)  # [batch_size, 4]

    norm = torch.norm(v, dim=2)  # [batch_size, 4]
    chan_gain = bk_val * hparams.beta_1 / norm + (1 - bk_val) * hparams.beta_2 / (norm ** 1.65)  # [batch_size, 4]

    snr = hparams.P_AVG * chan_gain / hparams.noise  # [batch_size, 4]
    se = torch.log2(1 + snr)  # [batch_size, 4]

    return -torch.mean(se)
