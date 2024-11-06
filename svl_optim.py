import optuna
import os
import logging
import torch
import pandas as pd
import numpy as np
import wandb  # wandb 추가
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from model import Net, MPL
from datasets import SvlDataset

# wandb 프로젝트 이름과 설정
wandb_project_name = "DL Based Wireless AP Optimise Using Blockage Aware Model"
random_seed = 42
lr = 2.5e-4

# Optuna 목적 함수 정의
def objective(trial):
    # 랜덤 시드, 학습률, 배치 크기 값을 설정
    hidden_N = 2**trial.suggest_int('hidden_N', 10, 14)
    hidden_L = trial.suggest_int('hidden_L', 4, 16)
    batch_size = trial.suggest_categorical('batch_size', [2**11, 2**12, 2**13, 2**14])

    # wandb 초기화 및 설정
    wandb.init(project=wandb_project_name, config={
        "hidden_N": hidden_N,
        "hidden_L": hidden_L,
        "batch_size": batch_size,
    })

    # 재현성을 위한 시드 설정
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 데이터 준비
    df = pd.concat([pd.read_csv('data/data1.csv'), pd.read_csv('data/data2.csv')])
    x = df.iloc[:, :12].values
    y = df.iloc[:, 12:].values

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y)
    x_train, x_val, y_train, y_val = train_test_split(x_scaled, y_scaled, test_size=0.25, random_state=random_seed)

    # 데이터셋 및 DataLoader 생성
    train_dataset = SvlDataset(x_train, y_train, dtype=torch.float32).to(device)
    val_dataset = SvlDataset(x_val, y_val, dtype=torch.float32).to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 모델, 옵티마이저, 손실 함수 설정
    model = Net(train_dataset.x.shape[1], hidden_N, hidden_L).to(device)(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 학습 루프
    for epoch in range(1000):
        model.train()
        total_loss = []
        for x_batch, y_batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = torch.nn.functional.mse_loss(y_pred, y_batch)
            loss.backward()
            total_loss.append(loss.item())
            optimizer.step()

        # 학습 손실
        train_loss = np.mean(total_loss)

        # 검증 루프
        model.eval()
        total_val_loss = []
        y_val_preds, y_vals = [], []
        with torch.no_grad():
            for x_val_batch, y_val_batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                y_val_pred = model(x_val_batch)
                val_loss = torch.nn.functional.mse_loss(y_val_pred, y_val_batch)
                total_val_loss.append(val_loss.item())
                y_val_preds.append(y_val_pred.cpu())
                y_vals.append(y_val_batch.cpu())

        # 검증 손실 및 R2 점수 계산 후 wandb 로그 기록
        val_loss = np.mean(total_val_loss)
        y_val_preds = torch.cat(y_val_preds).numpy()
        y_vals = torch.cat(y_vals).numpy()
        r2 = r2_score(y_vals, y_val_preds)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_r2": r2})

        # Optuna에 R2 점수 보고 및 prunning
        trial.report(r2, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

    wandb.finish()
    return r2  # R2 값 최대화를 목표로 설정

# 스터디 실행
if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    for _ in tqdm(range(50), desc="Optuna Trials"):
        study.optimize(objective, n_trials=1)

    # 최적화된 결과 출력
    best_trial = study.best_trial
    print(f"Best R2 Score: {best_trial.value}")
    print(f"Best Parameters: {best_trial.params}")
