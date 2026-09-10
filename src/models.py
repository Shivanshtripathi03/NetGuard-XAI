"""
NetGuard-XAI Model Architecture Definitions & Training Utilities
Includes XGBoost wrapper, PyTorch Autoencoder for anomaly detection,
and PyTorch LSTM for temporal behavioral analysis.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------
# 1. PyTorch Autoencoder (Unsupervised Anomaly Detector)
# ---------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def train_autoencoder(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=7, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_val,) in val_loader:
                x_val = x_val.to(device)
                recon = model(x_val)
                loss = criterion(recon, x_val)
                val_loss += loss.item() * len(x_val)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[AE] Early stopping at epoch {epoch+1}. Best Val Loss: {best_loss:.6f}")
                break
                
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss

# ---------------------------------------------------------
# 2. PyTorch LSTM (Temporal Behavioral Analyzer)
# ---------------------------------------------------------
class LSTMBehaviorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMBehaviorModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take output of last time step
        last_out = lstm_out[:, -1, :]
        prob = self.fc(last_out)
        return prob.squeeze(-1)

def train_lstm(model, train_loader, val_loader, epochs=40, lr=1e-3, patience=6, device='cpu'):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val)
                loss = criterion(preds, y_val)
                val_loss += loss.item() * len(x_val)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[LSTM] Early stopping at epoch {epoch+1}. Best Val Loss: {best_loss:.6f}")
                break
                
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss

# ---------------------------------------------------------
# 3. Helper for Constructing Sequence Batches
# ---------------------------------------------------------
def create_sequences(X, y, seq_length=10):
    """
    Constructs rolling window sequences of length `seq_length`.
    Sequence label is 1 if any event in the window is an attack (or last event is attack).
    """
    num_samples = len(X) - seq_length + 1
    if num_samples <= 0:
        raise ValueError("Dataset smaller than sequence length!")
        
    X_seq = np.zeros((num_samples, seq_length, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros(num_samples, dtype=np.float32)
    
    for i in range(num_samples):
        X_seq[i] = X[i : i + seq_length]
        # Sequence is labeled anomalous if last event is attack
        y_seq[i] = y[i + seq_length - 1]
        
    return X_seq, y_seq
