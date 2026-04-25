import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import os
import pickle
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy

class SensorDataProcessor:
    """Class for processing LiDAR, Radar, and Camera sensor data"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = None
        self.sensor_types = [
            'LIDAR_TOP', 
            'RADAR_FRONT', 
            'RADAR_FRONT_LEFT', 
            'RADAR_FRONT_RIGHT',
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT'
        ]
        
        # Features for each sensor
        self.sensor_features = {
            'LIDAR_TOP': [
                'lidar_point_distance_distribution_faulty',
                'lidar_obstacle_sector_distribution_faulty',
                'lidar_points_sector_distribution_faulty',
                'lidar_obstacle_count_faulty',
                'lidar_obstacle_distance_distribution_faulty'
            ],
            'RADAR_FRONT': [
                'radarFront_point_distance_distribution_faulty',
                'radarFront_obstacle_sector_distribution_faulty',
                'radarFront_points_sector_distribution_faulty',
                'radarFront_obstacle_count_faulty',
                'radarFront_obstacle_distance_distribution_faulty',
                'radarFront_velocity_x_distribution_faulty',
                'radarFront_velocity_y_distribution_faulty'
            ],
            'RADAR_FRONT_LEFT': [
                'radarFront_point_distance_distribution_faulty',
                'radarFront_obstacle_sector_distribution_faulty',
                'radarFront_points_sector_distribution_faulty',
                'radarFront_obstacle_count_faulty',
                'radarFront_obstacle_distance_distribution_faulty',
                'radarFront_velocity_x_distribution_faulty',
                'radarFront_velocity_y_distribution_faulty'
            ],
            'RADAR_FRONT_RIGHT': [
                'radarFront_point_distance_distribution_faulty',
                'radarFront_obstacle_sector_distribution_faulty',
                'radarFront_points_sector_distribution_faulty',
                'radarFront_obstacle_count_faulty',
                'radarFront_obstacle_distance_distribution_faulty',
                'radarFront_velocity_x_distribution_faulty',
                'radarFront_velocity_y_distribution_faulty'
            ],
            'CAM_FRONT': [
                'camera_obstacle_distance_distribution_faulty'
            ],
            'CAM_FRONT_LEFT': [
                'camera_obstacle_distance_distribution_faulty'
            ],
            'CAM_FRONT_RIGHT': [
                'camera_obstacle_distance_distribution_faulty'
            ]
        }
        
        # Feature dimensions
        self.feature_dims = {
            'lidar_point_distance_distribution_faulty': 20,
            'lidar_obstacle_sector_distribution_faulty': 60,
            'lidar_points_sector_distribution_faulty': 60,
            'lidar_obstacle_count_faulty': 1,
            'lidar_obstacle_distance_distribution_faulty': 20,
            'radarFront_point_distance_distribution_faulty': 50,
            'radarFront_obstacle_sector_distribution_faulty': 60,
            'radarFront_points_sector_distribution_faulty': 60,
            'radarFront_obstacle_count_faulty': 1,
            'radarFront_obstacle_distance_distribution_faulty': 50,
            'radarFront_velocity_x_distribution_faulty': 100,
            'radarFront_velocity_y_distribution_faulty': 100,
            'camera_obstacle_distance_distribution_faulty': 40
        }
        
        # Define feature weights for weighted majority voting
        self.feature_weights = {
            'LIDAR_TOP': {
                'lidar_point_distance_distribution_faulty': 4.0,    
                'lidar_obstacle_sector_distribution_faulty': 2.0,   
                'lidar_points_sector_distribution_faulty': 4.0,     
                'lidar_obstacle_count_faulty': 0.5,                 
                'lidar_obstacle_distance_distribution_faulty': 2.0  
            },
            'RADAR_FRONT': {
                'radarFront_obstacle_sector_distribution_faulty': 3.0,
                'radarFront_obstacle_distance_distribution_faulty': 2.5,
                'radarFront_points_sector_distribution_faulty': 3.0,
                'radarFront_point_distance_distribution_faulty': 2.0,
                'radarFront_obstacle_count_faulty': 1.0,
                'radarFront_velocity_x_distribution_faulty': 0.5,
                'radarFront_velocity_y_distribution_faulty': 0.5
            },
            'RADAR_FRONT_LEFT': {
                'radarFront_obstacle_sector_distribution_faulty': 3.0,
                'radarFront_obstacle_distance_distribution_faulty': 2.5,
                'radarFront_points_sector_distribution_faulty': 3.0,
                'radarFront_point_distance_distribution_faulty': 2.0,
                'radarFront_obstacle_count_faulty': 1.0,
                'radarFront_velocity_x_distribution_faulty': 0.5,
                'radarFront_velocity_y_distribution_faulty': 0.5
            },
            'RADAR_FRONT_RIGHT': {
                'radarFront_obstacle_sector_distribution_faulty': 3.0,
                'radarFront_obstacle_distance_distribution_faulty': 2.5,
                'radarFront_points_sector_distribution_faulty': 3.0,
                'radarFront_point_distance_distribution_faulty': 2.0,
                'radarFront_obstacle_count_faulty': 1.0,
                'radarFront_velocity_x_distribution_faulty': 0.5,
                'radarFront_velocity_y_distribution_faulty': 0.5
            },
            'CAM_FRONT': {
                'camera_obstacle_distance_distribution_faulty': 1.0
            },
            'CAM_FRONT_LEFT': {
                'camera_obstacle_distance_distribution_faulty': 1.0
            },
            'CAM_FRONT_RIGHT': {
                'camera_obstacle_distance_distribution_faulty': 1.0
            }
        }
        
        
        self.feature_thresholds = {
            'LIDAR_TOP': {
                'lidar_point_distance_distribution_faulty': 0.3,      
                'lidar_points_sector_distribution_faulty': 0.5,       
                'lidar_obstacle_sector_distribution_faulty': 0.5,     
                'lidar_obstacle_distance_distribution_faulty': 0.5,   
                'lidar_obstacle_count_faulty': 0.7,                  
            },
            'RADAR_FRONT': {
                'radarFront_obstacle_sector_distribution_faulty': 0.5,   
                'radarFront_points_sector_distribution_faulty': 0.5,      
                'radarFront_obstacle_distance_distribution_faulty': 0.5,  
                'radarFront_point_distance_distribution_faulty': 0.5,     
                'radarFront_obstacle_count_faulty': 0.5,                  
                'radarFront_velocity_x_distribution_faulty': 0.5,         
                'radarFront_velocity_y_distribution_faulty': 0.5,         
            },
            'RADAR_FRONT_LEFT': {
                'radarFront_obstacle_sector_distribution_faulty': 0.5,
                'radarFront_points_sector_distribution_faulty': 0.5,
                'radarFront_obstacle_distance_distribution_faulty': 0.5,
                'radarFront_point_distance_distribution_faulty': 0.5,
                'radarFront_obstacle_count_faulty': 0.5,
                'radarFront_velocity_x_distribution_faulty': 0.5,
                'radarFront_velocity_y_distribution_faulty': 0.5,
            },
            'RADAR_FRONT_RIGHT': {
                'radarFront_obstacle_sector_distribution_faulty': 0.5,
                'radarFront_points_sector_distribution_faulty': 0.5,
                'radarFront_obstacle_distance_distribution_faulty': 0.5,
                'radarFront_point_distance_distribution_faulty': 0.5,
                'radarFront_obstacle_count_faulty': 0.5,
                'radarFront_velocity_x_distribution_faulty': 0.5,
                'radarFront_velocity_y_distribution_faulty': 0.5,
            },
            
            'CAM_FRONT': {
                'camera_obstacle_distance_distribution_faulty': 0.55,  
            },
            'CAM_FRONT_LEFT': {
                'camera_obstacle_distance_distribution_faulty': 0.55,  
            },
            'CAM_FRONT_RIGHT': {
                'camera_obstacle_distance_distribution_faulty': 0.55,  
            }
        }
        
        # Threshold explanations
        self.threshold_explanations = {
            'LIDAR_TOP': {
                'high_sensitivity': '0.3 - Easy fault detection (high sensitivity)',
                'medium_sensitivity': '0.4 - Medium fault detection',
                'low_sensitivity': '0.6 - Difficult fault detection (low sensitivity)'
            },
            'RADAR': {
                'high_sensitivity': '0.5 - Relatively easy fault detection (considering noise)',
                'medium_sensitivity': '0.6 - Medium fault detection (higher threshold due to noise)',
                'low_sensitivity': '0.7 - Difficult fault detection (radar high noise)'
            },
            'CAMERA': {
                'high_sensitivity': '0.5 - Standard sensitivity for camera',
                'medium_sensitivity': '0.6 - Medium sensitivity for camera',
                'low_sensitivity': '0.7 - Low sensitivity for camera'
            }
        }
    
    def get_feature_threshold(self, sensor_type: str, feature_name: str) -> float:
        """Get threshold for a specific feature and sensor"""
        sensor_thresholds = self.feature_thresholds.get(sensor_type, {})
        return sensor_thresholds.get(feature_name, 0.5)
    
    def load_data(self) -> Dict:
        """Load data from JSON file"""
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        print(f"Data loaded from {self.json_path}")
        return self.data
    
    def parse_feature_value(self, value: Any) -> np.ndarray:
        """Parse feature value from various formats"""
        if isinstance(value, list):
            return np.array(value, dtype=np.float32)
        elif isinstance(value, (int, float)):
            return np.array([float(value)], dtype=np.float32)
        elif isinstance(value, dict):
            if 'values' in value:
                return np.array(value['values'], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)
    
    def process_sensor_data(self, sensor_type: str) -> List[Dict]:
        """Process data for a specific sensor"""
        if sensor_type not in self.data:
            print(f"Sensor {sensor_type} not found in data")
            return []
        
        sensor_data = self.data[sensor_type]
        records = []
        
        # Extract records
        for token, record_data in sensor_data.items():
            if isinstance(record_data, dict):
                # Handle different field names for camera sensors
                if sensor_type.startswith('CAM'):
                    token_key = 'Token' if 'Token' in record_data else 'token'
                    sample_key = 'sample_token'
                    timestamp_key = 'timestamp'
                    sensor_type_key = 'sensor_type'
                    is_faulty_key = 'is_faulty'
                    feature_vectors_key = 'feature_vectors'
                else:
                    token_key = 'token'
                    sample_key = 'sample_token'
                    timestamp_key = 'timestamp'
                    sensor_type_key = 'sensor_type'
                    is_faulty_key = 'is_faulty'
                    feature_vectors_key = 'feature_vectors'
                
                # Get is_faulty
                is_faulty = record_data.get(is_faulty_key)
                
                if is_faulty is None or is_faulty == '':
                    is_faulty_val = None
                else:
                    try:
                        is_faulty_val = int(is_faulty)
                    except (ValueError, TypeError):
                        is_faulty_val = None
                
                record = {
                    'token': record_data[token_key],
                    'sample_token': record_data.get(sample_key, ''),
                    'timestamp': record_data.get(timestamp_key, 0),
                    'is_faulty': is_faulty_val,
                    'sensor_type': sensor_type
                }
                
                # Extract all available fields
                for field in ['ego_pose_token', 'calibration_token', 'original_file', 
                             'prev', 'next', 'faulty_instance_token', 'interpolation_info', 'fault_type']:
                    if field in record_data:
                        record[field] = record_data[field]
                
                # Extract feature vectors
                feature_vectors = {}
                if feature_vectors_key in record_data:
                    for feature in self.sensor_features[sensor_type]:
                        if feature in record_data[feature_vectors_key]:
                            feature_data = record_data[feature_vectors_key][feature]
                            feature_vectors[feature] = self.parse_feature_value(feature_data)
                        else:
                            dim = self.feature_dims[feature]
                            feature_vectors[feature] = np.zeros(dim, dtype=np.float32)
                else:
                    for feature in self.sensor_features[sensor_type]:
                        dim = self.feature_dims[feature]
                        feature_vectors[feature] = np.zeros(dim, dtype=np.float32)
                
                record['feature_vectors'] = feature_vectors
                records.append(record)
        
        # Sort by timestamp
        records.sort(key=lambda x: x['timestamp'])
        
        print(f"Processed {len(records)} records for {sensor_type}")
        
        # Count test records
        test_records = [r for r in records if r['is_faulty'] is None]
        print(f"  Test records: {len(test_records)} (is_faulty=None)")
        
        return records
    
    def prepare_sequences(self, sensor_records: List[Dict], feature_name: str, 
                         sequence_length: int = 4) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare time sequences for a specific feature"""
        sequences = []
        labels = []
        tokens = []
        
        # Create sequences for each record
        for i in range(len(sensor_records)):
            current_sequence = []
            current_token = sensor_records[i]['token']
            
            # Collect previous records (up to 3 previous)
            for j in range(max(0, i - sequence_length + 1), i + 1):
                if feature_name in sensor_records[j]['feature_vectors']:
                    feature_vector = sensor_records[j]['feature_vectors'][feature_name]
                    dim = self.feature_dims[feature_name]
                    if len(feature_vector) != dim:
                        if len(feature_vector) < dim:
                            feature_vector = np.pad(feature_vector, (0, dim - len(feature_vector)), 'constant')
                        else:
                            feature_vector = feature_vector[:dim]
                    current_sequence.append(feature_vector)
                else:
                    dim = self.feature_dims[feature_name]
                    current_sequence.append(np.zeros(dim, dtype=np.float32))
            
            # Pad sequence if shorter than sequence_length
            while len(current_sequence) < sequence_length:
                dim = self.feature_dims[feature_name]
                current_sequence.insert(0, np.zeros(dim, dtype=np.float32))
            
            sequences.append(np.array(current_sequence))
            
            is_faulty = sensor_records[i]['is_faulty']
            labels.append(0 if is_faulty is None else is_faulty)
            tokens.append(current_token)
        
        sequences_array = np.array(sequences)
        labels_array = np.array(labels, dtype=np.float32)
        
        print(f"Prepared {len(sequences_array)} sequences for feature {feature_name}, shape: {sequences_array.shape}")
        return sequences_array, labels_array, tokens
    
    def split_data(self, sensor_records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Split data into training and testing sets based on is_faulty"""
        train_records = [r for r in sensor_records if r['is_faulty'] is not None]
        test_records = [r for r in sensor_records if r['is_faulty'] is None]
        
        print(f"Data split: {len(train_records)} training, {len(test_records)} testing records")
        
        if len(train_records) < 30:
            print(f"Warning: Only {len(train_records)} training records for sensor. Using 70/30 split.")
            total_records = len(sensor_records)
            train_size = int(0.7 * total_records)
            train_records = sensor_records[:train_size]
            test_records = sensor_records[train_size:]
        
        return train_records, test_records


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMModel(nn.Module):
    """LSTM model for fault detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                 dropout: float = 0.3, is_radar: bool = False, is_camera: bool = False):
        super(LSTMModel, self).__init__()
        
        if is_radar:
            dropout = 0.35
        elif is_camera:
            dropout = 0.3  
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        x = self.dropout(last_hidden)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class RNNModel(nn.Module):
    """RNN model for fault detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, is_radar: bool = False, is_camera: bool = False):
        super(RNNModel, self).__init__()
        
        if is_radar:
            dropout = 0.35
        elif is_camera:
            dropout = 0.3  
        
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        last_hidden = rnn_out[:, -1, :]
        
        x = self.dropout(last_hidden)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class FaultDetectionModel:
    """Class for creating and training fault detection models"""
    
    def __init__(self, model_type: str = 'lstm', device: str = None):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def build_model(self, input_dim: int, is_radar: bool = False, is_camera: bool = False) -> nn.Module:
        """Build model based on model_type"""
        if self.model_type == 'lstm':
            return LSTMModel(input_dim=input_dim, is_radar=is_radar, is_camera=is_camera)
        else:
            return RNNModel(input_dim=input_dim, is_radar=is_radar, is_camera=is_camera)
    
    def train_feature_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                           feature_name: str, sensor_type: str,
                           epochs: int = 50, batch_size: int = 32,
                           learning_rate: float = 0.001,
                           use_kfold: bool = False) -> nn.Module:
        """Train model for a specific feature"""
        
        if len(X_train) < 10:
            print(f"Warning: Insufficient training data for {feature_name}. Skipping.")
            return None
        
        is_radar = sensor_type.startswith('RADAR')
        is_camera = sensor_type.startswith('CAM')
        
        # Normalize data
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
        self.scalers[(sensor_type, feature_name)] = scaler
        
        
        if use_kfold and is_radar:  
            print(f"Applying 5-fold cross validation for {sensor_type} - {feature_name}")
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_models = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled), 1):
                print(f"\nFold {fold}/5:")
                
                X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                train_dataset = TimeSeriesDataset(X_fold_train, y_fold_train)
                val_dataset = TimeSeriesDataset(X_fold_val, y_fold_val)
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                input_dim = X_fold_train.shape[2]
                model = self.build_model(input_dim, is_radar=is_radar, is_camera=is_camera)
                model.to(self.device)
                
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
                
                model.train()
                best_val_loss = float('inf')
                best_model_state = None
                
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0.0
                    train_correct = 0
                    train_total = 0
                    
                    for batch_X, batch_y in train_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device).unsqueeze(1)
                        
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        
                        probabilities = torch.sigmoid(outputs)
                        predicted = (probabilities > 0.5).float()
                        train_total += batch_y.size(0)
                        train_correct += (predicted == batch_y).sum().item()
                    
                    avg_train_loss = train_loss / len(train_loader)
                    train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
                    
                    model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X = batch_X.to(self.device)
                            batch_y = batch_y.to(self.device).unsqueeze(1)
                            
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            val_loss += loss.item()
                            
                            probabilities = torch.sigmoid(outputs)
                            predicted = (probabilities > 0.5).float()
                            val_total += batch_y.size(0)
                            val_correct += (predicted == batch_y).sum().item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                    val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
                    
                    scheduler.step(avg_val_loss)
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_state = deepcopy(model.state_dict())
                    
                    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"  Epoch {epoch+1}/{epochs}: "
                              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                
                fold_models.append(best_model_state)
                print(f"  Fold {fold} completed. Best val loss: {best_val_loss:.4f}")
            
            print("\nAveraging models from all folds...")
            avg_model_state = {}
            
            for key in fold_models[0].keys():
                param_sum = torch.zeros_like(fold_models[0][key])
                for fold_state in fold_models:
                    param_sum += fold_state[key]
                avg_model_state[key] = param_sum / len(fold_models)
            
            input_dim = X_train_scaled.shape[2]
            final_model = self.build_model(input_dim, is_radar=is_radar, is_camera=is_camera)
            final_model.load_state_dict(avg_model_state)
            final_model.to(self.device)
            
            print("Final training on full dataset...")
            full_dataset = TimeSeriesDataset(X_train_scaled, y_train)
            full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(final_model.parameters(), lr=learning_rate/2)
            final_model.train()
            
            for epoch in range(epochs//2):
                epoch_loss = 0.0
                correct = 0
                total = 0
                
                for batch_X, batch_y in full_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device).unsqueeze(1)
                    
                    optimizer.zero_grad()
                    outputs = final_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    probabilities = torch.sigmoid(outputs)
                    predicted = (probabilities > 0.5).float()
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                
                avg_loss = epoch_loss / len(full_loader)
                accuracy = 100 * correct / total if total > 0 else 0
                
                if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == (epochs//2) - 1:
                    print(f"  Final training epoch {epoch+1}/{epochs//2}: "
                          f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
            
            model = final_model
            
        else:
            
            dataset = TimeSeriesDataset(X_train_scaled, y_train)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            input_dim = X_train_scaled.shape[2]
            model = self.build_model(input_dim, is_radar=is_radar, is_camera=is_camera)
            model.to(self.device)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
            
            model.train()
            best_loss = float('inf')
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                correct = 0
                total = 0
                
                for batch_X, batch_y in dataloader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device).unsqueeze(1)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    probabilities = torch.sigmoid(outputs)
                    predicted = (probabilities > 0.5).float()
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                
                avg_loss = epoch_loss / len(dataloader)
                accuracy = 100 * correct / total if total > 0 else 0
                
                scheduler.step(avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                
                if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                          f"Accuracy: {accuracy:.2f}%, LR: {current_lr:.6f}")
        
        self.models[(sensor_type, feature_name)] = model
        print(f"Training completed for {feature_name}")
        
        return model
    
    def predict_feature(self, X_test: np.ndarray, feature_name: str, 
                       sensor_type: str, threshold: float = None, 
                       batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, float]:
        """Make predictions for a specific feature"""
        if (sensor_type, feature_name) not in self.models:
            raise ValueError(f"Model for feature {feature_name} and sensor {sensor_type} not trained")
        
        model = self.models[(sensor_type, feature_name)]
        scaler = self.scalers[(sensor_type, feature_name)]
        
        if threshold is None:
            threshold = 0.5
        
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        dataset = TimeSeriesDataset(X_test_scaled, np.zeros(len(X_test_scaled)))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        model.eval()
        probabilities_list = []
        
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                probabilities = torch.sigmoid(outputs)
                probabilities_list.extend(probabilities.cpu().numpy())
        
        probabilities_array = np.array(probabilities_list).flatten()
        predictions_binary = (probabilities_array > threshold).astype(int)
        
        return probabilities_array, predictions_binary, threshold
    
    def save_model(self, model: nn.Module, feature_name: str, sensor_type: str, 
                  model_path: str):
        """Save model to file"""
        model_filename = f"{sensor_type}_{feature_name}_{self.model_type}.pth"
        model_filepath = os.path.join(model_path, model_filename)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_name': feature_name,
            'sensor_type': sensor_type,
            'model_type': self.model_type,
            'input_dim': model.lstm.input_size if self.model_type == 'lstm' else model.rnn.input_size,
        }, model_filepath)
        
        print(f"Model saved to {model_filepath}")
    
    def load_model(self, model_path: str, feature_name: str, sensor_type: str, 
                  input_dim: int) -> nn.Module:
        """Load model from file"""
        model_filename = f"{sensor_type}_{feature_name}_{self.model_type}.pth"
        model_filepath = os.path.join(model_path, model_filename)
        
        checkpoint = torch.load(model_filepath, map_location=self.device)
        model = self.build_model(input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        self.models[(sensor_type, feature_name)] = model
        
        return model


class FaultDetectionSystem:
    """Complete fault detection system"""
    
    def __init__(self, json_path: str, model_type: str = 'lstm'):
        self.processor = SensorDataProcessor(json_path)
        self.model = FaultDetectionModel(model_type)
        self.predictions = {}
        
        # Paths for saving
        self.result_path = r"C:\PATH\LSTM\Result"
        self.model_path = r"C:\PATH\LSTM\Model"
        
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
    
    def process_all_sensors(self) -> Dict[str, List[Dict]]:
        """Process all sensors"""
        self.processor.load_data()
        all_sensor_data = {}
        
        for sensor_type in self.processor.sensor_types:
            print(f"\n{'='*50}")
            print(f"Processing sensor: {sensor_type}")
            print(f"{'='*50}")
            
            sensor_records = self.processor.process_sensor_data(sensor_type)
            if sensor_records:
                all_sensor_data[sensor_type] = sensor_records
        
        return all_sensor_data
    
    def train_sensor_models(self, sensor_type: str, sensor_records: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """Train models for a sensor"""
        print(f"\nTraining models for sensor: {sensor_type}")
        
        is_radar = sensor_type.startswith('RADAR')
        is_lidar = sensor_type == 'LIDAR_TOP'
        is_camera = sensor_type.startswith('CAM')
        
        train_records, test_records = self.processor.split_data(sensor_records)
        
        if len(train_records) > 15:
            for i in range(min(15, len(train_records))):
                train_records[i]['is_faulty'] = 0
        
        print(f"Training records: {len(train_records)} (with labels)")
        print(f"Test records: {len(test_records)} (without labels)")
        
        print(f"\nFeature thresholds for {sensor_type}:")
        for feature_name in self.processor.sensor_features[sensor_type]:
            threshold = self.processor.get_feature_threshold(sensor_type, feature_name)
            print(f"  {feature_name}: {threshold}")
        
        feature_predictions = {}
        
        for feature_name in self.processor.sensor_features[sensor_type]:
            print(f"\nTraining model for feature: {feature_name}")
            
            X_train, y_train, train_tokens = self.processor.prepare_sequences(
                train_records, feature_name
            )
            
            if len(X_train) == 0:
                print(f"Warning: No training data for feature {feature_name}")
                continue
            
            
            use_kfold = is_radar  
            if use_kfold:
                print(f"Using 5-fold cross validation for radar")
            elif is_camera:
                print(f"Using standard training for camera (no K-fold)")
            
            trained_model = self.model.train_feature_model(
                X_train, y_train, feature_name, sensor_type,
                use_kfold=use_kfold
            )
            
            if trained_model is None:
                print(f"Skipping {feature_name} due to insufficient data")
                continue
            
            self.model.save_model(trained_model, feature_name, sensor_type, self.model_path)
            
            X_test, y_test, test_tokens = self.processor.prepare_sequences(
                test_records, feature_name
            )
            
            if len(X_test) == 0:
                print(f"Warning: No test data for feature {feature_name}")
                continue
            
            feature_threshold = self.processor.get_feature_threshold(sensor_type, feature_name)
            
            probabilities, predictions_binary, used_threshold = self.model.predict_feature(
                X_test, feature_name, sensor_type, threshold=feature_threshold
            )
            
            feature_predictions[feature_name] = {
                'tokens': test_tokens,
                'predictions': predictions_binary,
                'probabilities': probabilities,
                'actual': y_test,
                'threshold': used_threshold
            }
            
            if len(y_train) > 0:
                val_prob, val_pred, _ = self.model.predict_feature(
                    X_train[:50], feature_name, sensor_type, threshold=feature_threshold
                )
                val_acc = np.mean(val_pred == y_train[:50])
                print(f"Feature {feature_name} validation accuracy (threshold={feature_threshold}): {val_acc:.4f}")
        
        return feature_predictions, test_records
    
    def weighted_majority_vote(self, feature_predictions: Dict, test_records: List[Dict], 
                               sensor_type: str) -> Tuple[List[Dict], np.ndarray, Dict]:
        """Perform weighted majority voting between different feature predictions"""
        if not feature_predictions:
            return [], np.array([]), {}
        
        is_camera = sensor_type.startswith('CAM')
        
        # For camera sensors with single feature, no voting needed
        if is_camera and len(feature_predictions) == 1:
            feature_name = list(feature_predictions.keys())[0]
            pred_data = feature_predictions[feature_name]
            
            test_results = []
            final_predictions = []
            feature_votes = {}
            
            test_records_dict = {r['token']: r for r in test_records}
            
            for i, token in enumerate(pred_data['tokens']):
                if token in test_records_dict:
                    record = test_records_dict[token]
                    
                    prediction = int(pred_data['predictions'][i])
                    probability = float(pred_data['probabilities'][i])
                    threshold = float(pred_data.get('threshold', 0.5))
                    
                    result = {
                        'token': token,
                        'sample_token': record.get('sample_token', ''),
                        'timestamp': record.get('timestamp', 0),
                        'predicted_faulty': prediction,
                        'actual_faulty': None,
                        'sensor_type': record['sensor_type'],
                        'feature_predictions': {
                            feature_name: {
                                'prediction': prediction,
                                'probability': probability,
                                'threshold': threshold
                            }
                        },
                        'feature_thresholds': {feature_name: threshold},
                        'weighted_voting_stats': {
                            'weighted_fault_votes': float(prediction),
                            'weighted_healthy_votes': float(1 - prediction),
                            'total_weight': 1.0,
                            'weighted_probability': float(probability),
                            'decision_basis': 'single_feature_camera'
                        }
                    }
                    
                    for field in ['ego_pose_token', 'calibration_token', 'original_file', 
                                'prev', 'next', 'faulty_instance_token']:
                        if field in record:
                            result[field] = record[field]
                    
                    test_results.append(result)
                    final_predictions.append(prediction)
                    
                    feature_votes[token] = {
                        'single_feature': {
                            'name': feature_name,
                            'prediction': prediction,
                            'probability': probability,
                            'threshold': threshold
                        },
                        'majority': prediction
                    }
            
            return test_results, np.array(final_predictions), feature_votes
        
        # Original weighted majority voting for sensors with multiple features
        all_predictions_dict = {}
        feature_details = {}
        
        for feature_name, pred_data in feature_predictions.items():
            for i, token in enumerate(pred_data['tokens']):
                if token not in all_predictions_dict:
                    all_predictions_dict[token] = {
                        'predictions': [],
                        'probabilities': [],
                        'feature_details': {},
                        'weighted_details': {},
                        'thresholds': {}
                    }
                all_predictions_dict[token]['predictions'].append(pred_data['predictions'][i])
                all_predictions_dict[token]['probabilities'].append(pred_data['probabilities'][i])
                all_predictions_dict[token]['feature_details'][feature_name] = {
                    'prediction': int(pred_data['predictions'][i]),
                    'probability': float(pred_data['probabilities'][i]),
                    'threshold': float(pred_data.get('threshold', 0.5))
                }
                all_predictions_dict[token]['thresholds'][feature_name] = pred_data.get('threshold', 0.5)
        
        final_results = []
        final_predictions = []
        feature_votes = {}
        
        test_records_dict = {r['token']: r for r in test_records}
        feature_weights = self.processor.feature_weights.get(sensor_type, {})
        
        print(f"\nWeighted Majority Voting for {sensor_type}:")
        print("Feature Weights and Thresholds:")
        for feature_name, weight in feature_weights.items():
            threshold = self.processor.get_feature_threshold(sensor_type, feature_name)
            print(f"  {feature_name}: Weight={weight}, Threshold={threshold}")
        
        for token, data in all_predictions_dict.items():
            if token in test_records_dict:
                record = test_records_dict[token]
                
                weighted_fault_votes = 0.0
                weighted_healthy_votes = 0.0
                total_weight = 0.0
                
                for feature_name, pred_data in data['feature_details'].items():
                    weight = feature_weights.get(feature_name, 1.0)
                    total_weight += weight
                    
                    if pred_data['prediction'] == 1:
                        weighted_fault_votes += weight
                    else:
                        weighted_healthy_votes += weight
                    
                    data['weighted_details'][feature_name] = {
                        'prediction': pred_data['prediction'],
                        'probability': pred_data['probability'],
                        'weight': weight,
                        'threshold': pred_data['threshold'],
                        'weighted_vote': weight if pred_data['prediction'] == 1 else 0
                    }
                
                weighted_prob_sum = 0.0
                weight_sum = 0.0
                for feature_name, pred_data in data['feature_details'].items():
                    weight = feature_weights.get(feature_name, 1.0)
                    weighted_prob_sum += weight * pred_data['probability']
                    weight_sum += weight
                
                weighted_probability = weighted_prob_sum / weight_sum if weight_sum > 0 else 0.0
                
                if weighted_fault_votes > weighted_healthy_votes:
                    majority = 1
                elif weighted_fault_votes < weighted_healthy_votes:
                    majority = 0
                else:
                    majority = 1 if weighted_probability > 0.5 else 0
                
                feature_votes_list = []
                for feature_name, details in data['weighted_details'].items():
                    feature_votes_list.append({
                        'feature': feature_name,
                        'prediction': details['prediction'],
                        'probability': details['probability'],
                        'weight': details['weight'],
                        'threshold': details['threshold'],
                        'weighted_vote': details['weighted_vote']
                    })
                
                result = {
                    'token': token,
                    'sample_token': record.get('sample_token', ''),
                    'timestamp': record.get('timestamp', 0),
                    'predicted_faulty': int(majority),
                    'actual_faulty': None,
                    'sensor_type': record['sensor_type'],
                    'feature_predictions': data['feature_details'],
                    'feature_weights': feature_weights,
                    'feature_thresholds': data['thresholds'],
                    'weighted_details': data['weighted_details'],
                    'feature_votes': feature_votes_list,
                    'weighted_voting_stats': {
                        'weighted_fault_votes': float(weighted_fault_votes),
                        'weighted_healthy_votes': float(weighted_healthy_votes),
                        'total_weight': float(total_weight),
                        'weighted_probability': float(weighted_probability),
                        'decision_basis': 'weighted_votes' if weighted_fault_votes != weighted_healthy_votes else 'probability_tie_break'
                    }
                }
                
                for field in ['ego_pose_token', 'calibration_token', 'original_file', 
                            'prev', 'next', 'faulty_instance_token']:
                    if field in record:
                        result[field] = record[field]
                
                final_results.append(result)
                final_predictions.append(majority)
                
                feature_votes[token] = {
                    'votes': data['predictions'],
                    'probabilities': data['probabilities'],
                    'weights': [feature_weights.get(f, 1.0) for f in data['feature_details'].keys()],
                    'thresholds': [data['thresholds'].get(f, 0.5) for f in data['feature_details'].keys()],
                    'majority': majority,
                    'feature_details': data['feature_details'],
                    'weighted_details': data['weighted_details'],
                    'weighted_fault_votes': weighted_fault_votes,
                    'weighted_healthy_votes': weighted_healthy_votes
                }
        
        return final_results, np.array(final_predictions), feature_votes
    
    def calculate_metrics(self, predictions: np.ndarray, actual: np.ndarray) -> Dict:
        """Calculate evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        if len(predictions) == 0 or len(actual) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'confusion_matrix': [[0, 0], [0, 0]]
            }
        
        min_len = min(len(predictions), len(actual))
        predictions = predictions[:min_len]
        actual = actual[:min_len]
        
        predictions = predictions.astype(int)
        actual = actual.astype(int)
        
        unique_classes = np.unique(actual)
        if len(unique_classes) < 2:
            accuracy = np.mean(predictions == actual)
            precision_val = 0.0 if np.sum(predictions) == 0 else float(np.sum((predictions == 1) & (actual == 1)) / np.sum(predictions))
            recall_val = 0.0 if np.sum(actual) == 0 else float(np.sum((predictions == 1) & (actual == 1)) / np.sum(actual))
            f1_val = 0.0 if (precision_val + recall_val) == 0 else 2 * precision_val * recall_val / (precision_val + recall_val)
            
            cm = confusion_matrix(actual, predictions)
            if cm.shape == (1, 1):
                cm = [[cm[0, 0], 0], [0, 0]]
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision_val),
                'recall': float(recall_val),
                'f1_score': float(f1_val),
                'confusion_matrix': cm.tolist()
            }
        
        try:
            accuracy = accuracy_score(actual, predictions)
            precision = precision_score(actual, predictions, zero_division=0)
            recall = recall_score(actual, predictions, zero_division=0)
            f1 = f1_score(actual, predictions, zero_division=0)
            cm = confusion_matrix(actual, predictions)
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': cm.tolist()
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'confusion_matrix': [[0, 0], [0, 0]]
            }
    
    def run_detection(self) -> Dict:
        """Run complete fault detection system"""
        print("Starting fault detection system using LSTM/RNN")
        print("="*80)
        print("LiDAR: Standard training with dropout=0.3, Weighted voting")
        print("Radar: 5-fold cross validation with dropout=0.35, Weighted voting")
        print("Camera: Standard training with dropout=0.3, Single feature (no voting needed)")
        print("Different thresholds for each feature based on sensitivity and noise")
        print("="*80)
        
        all_sensor_data = self.process_all_sensors()
        
        final_results = {}
        
        for sensor_type, sensor_records in all_sensor_data.items():
            print(f"\n{'='*80}")
            print(f"Fault detection for sensor: {sensor_type}")
            print(f"{'='*80}")
            
            is_radar = sensor_type.startswith('RADAR')
            is_lidar = sensor_type == 'LIDAR_TOP'
            is_camera = sensor_type.startswith('CAM')
            
            if is_radar:
                print("Radar sensor detected - Using 5-fold cross validation with dropout=0.35")
                print("Different thresholds per feature (higher thresholds for noise reduction)")
                print("Weighted majority voting with feature importance weights")
            elif is_lidar:
                print("LiDAR sensor detected - Using standard training with dropout=0.3")
                print("Different thresholds per feature (lower thresholds for high sensitivity features)")
                print("Weighted majority voting with customizable feature weights")
            elif is_camera:
                print(f"Camera sensor detected - Using standard training")
                print("Single feature: camera_obstacle_distance_distribution_faulty (dim=40)")
                print("No K-fold cross validation - Standard training only")
                print(f"Dropout: 0.3, Threshold: 0.5")
                print("No weighted voting needed (single feature decision)")
            
            print(f"\nConfigured thresholds for {sensor_type}:")
            for feature_name in self.processor.sensor_features[sensor_type]:
                threshold = self.processor.get_feature_threshold(sensor_type, feature_name)
                print(f"  {feature_name}: {threshold}")
            
            feature_predictions, test_records = self.train_sensor_models(sensor_type, sensor_records)
            
            if not feature_predictions:
                print(f"No models trained for {sensor_type}")
                final_results[sensor_type] = {
                    'results': [],
                    'metrics': self.calculate_metrics(np.array([]), np.array([])),
                    'feature_accuracies': {}
                }
                continue
            
            test_results, final_predictions, feature_votes = self.weighted_majority_vote(
                feature_predictions, test_records, sensor_type
            )
            
            if len(test_results) == 0:
                print(f"No predictions for {sensor_type}")
                final_results[sensor_type] = {
                    'results': [],
                    'metrics': self.calculate_metrics(np.array([]), np.array([])),
                    'feature_accuracies': {}
                }
                continue
            
            fault_count = np.sum(final_predictions)
            total_count = len(final_predictions)
            fault_percentage = (fault_count / total_count) * 100 if total_count > 0 else 0
            
            feature_stats = {}
            for feature_name, pred_data in feature_predictions.items():
                preds = pred_data['predictions']
                feature_fault_count = np.sum(preds)
                feature_total = len(preds)
                threshold = pred_data.get('threshold', 0.5)
                feature_stats[feature_name] = {
                    'faulty_count': int(feature_fault_count),
                    'total_count': int(feature_total),
                    'faulty_percentage': float((feature_fault_count / feature_total) * 100) if feature_total > 0 else 0.0,
                    'threshold': float(threshold)
                }
            
            final_results[sensor_type] = {
                'results': test_results,
                'predictions_summary': {
                    'total_test_records': total_count,
                    'predicted_faulty': int(fault_count),
                    'predicted_healthy': int(total_count - fault_count),
                    'faulty_percentage': float(fault_percentage)
                },
                'feature_statistics': feature_stats,
                'feature_votes_sample': {k: v for i, (k, v) in enumerate(feature_votes.items()) if i < 5},
                'feature_weights': self.processor.feature_weights.get(sensor_type, {}),
                'feature_thresholds': {name: self.processor.get_feature_threshold(sensor_type, name) 
                                      for name in self.processor.sensor_features[sensor_type]}
            }
            
            print(f"\nResults for {sensor_type}:")
            print(f"  - Total test records: {total_count}")
            print(f"  - Predicted faulty: {fault_count} ({fault_percentage:.2f}%)")
            print(f"  - Predicted healthy: {total_count - fault_count}")
            
            print(f"\n  Feature statistics (with thresholds and weights):")
            for feature_name, stats in feature_stats.items():
                weight = self.processor.feature_weights.get(sensor_type, {}).get(feature_name, 1.0)
                threshold = stats['threshold']
                print(f"    - {feature_name}: {stats['faulty_count']}/{stats['total_count']} faulty "
                      f"({stats['faulty_percentage']:.1f}%), "
                      f"Threshold: {threshold}, Weight: {weight}")
        
        self.save_results(final_results)
        
        return final_results
    
    def save_results(self, results: Dict):
        """Save results to JSON file with all details"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"fault_detection_results_{timestamp}.json"
        result_filepath = os.path.join(self.result_path, result_filename)
        
        output_data = {}
        for sensor_type, sensor_data in results.items():
            output_data[sensor_type] = {
                'results': sensor_data['results'],
                'predictions_summary': sensor_data.get('predictions_summary', {}),
                'feature_statistics': sensor_data.get('feature_statistics', {}),
                'feature_votes_sample': sensor_data.get('feature_votes_sample', {}),
                'feature_weights': sensor_data.get('feature_weights', {}),
                'feature_thresholds': sensor_data.get('feature_thresholds', {}),
                'metadata': {
                    'total_records': len(sensor_data['results']),
                    'sensor_type': sensor_type,
                    'processing_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'model_type': self.model.model_type,
                    'is_radar': sensor_type.startswith('RADAR'),
                    'is_camera': sensor_type.startswith('CAM'),
                    'training_method': '5-fold CV' if sensor_type.startswith('RADAR') else 'Standard',
                    'dropout_rate': 0.35 if sensor_type.startswith('RADAR') else 0.3,
                    'voting_method': 'Single feature' if sensor_type.startswith('CAM') else 'Weighted Majority Vote',
                    'threshold_strategy': 'Different thresholds per feature based on sensitivity and noise'
                }
            }
        
        with open(result_filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {result_filepath}")
        
        self.save_simplified_results(results, timestamp)
        self.save_summary(results, timestamp)
    
    def save_simplified_results(self, results: Dict, timestamp: str):
        """Save simplified results for quick viewing"""
        simplified_filename = f"simplified_results_{timestamp}.json"
        simplified_filepath = os.path.join(self.result_path, simplified_filename)
        
        simplified_data = {}
        for sensor_type, sensor_data in results.items():
            sensor_results = []
            for result in sensor_data['results']:
                simplified_result = {
                    'token': result['token'],
                    'sample_token': result.get('sample_token', ''),
                    'timestamp': result.get('timestamp', 0),
                    'predicted_faulty': result['predicted_faulty'],
                    'sensor_type': result['sensor_type'],
                    'feature_predictions': result.get('feature_predictions', {}),
                    'feature_votes': result.get('feature_votes', []),
                    'feature_thresholds': result.get('feature_thresholds', {}),
                    'weighted_voting_stats': result.get('weighted_voting_stats', {})
                }
                sensor_results.append(simplified_result)
            
            simplified_data[sensor_type] = {
                'results': sensor_results,
                'summary': sensor_data.get('predictions_summary', {}),
                'feature_weights': sensor_data.get('feature_weights', {}),
                'feature_thresholds': sensor_data.get('feature_thresholds', {})
            }
        
        with open(simplified_filepath, 'w') as f:
            json.dump(simplified_data, f, indent=2, default=str)
        
        print(f"Simplified results saved to: {simplified_filepath}")
    
    def save_summary(self, results: Dict, timestamp: str):
        """Save results summary"""
        summary_filename = f"results_summary_{timestamp}.txt"
        summary_filepath = os.path.join(self.result_path, summary_filename)
        
        with open(summary_filepath, 'w') as f:
            f.write("Fault Detection Results Summary\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: {self.model.model_type.upper()}\n")
            f.write("Threshold Strategy: Different thresholds per feature based on sensitivity and noise\n")
            f.write("="*80 + "\n\n")
            
            f.write("Sensor Configuration:\n")
            f.write("-"*40 + "\n")
            f.write("LiDAR: Standard training, dropout=0.3, Weighted voting\n")
            f.write("Radar: 5-fold CV, dropout=0.35, Weighted voting\n")
            f.write("Camera: Standard training, dropout=0.3, Single feature (no voting needed)\n")
            f.write("Camera Config: threshold=0.5, no K-fold cross validation\n")
            f.write("="*80 + "\n\n")
            
            total_tests = 0
            total_faulty_predicted = 0
            
            for sensor_type, sensor_data in results.items():
                if 'predictions_summary' in sensor_data:
                    summary = sensor_data['predictions_summary']
                    total_records = summary.get('total_test_records', 0)
                    faulty_predicted = summary.get('predicted_faulty', 0)
                    
                    is_radar = sensor_type.startswith('RADAR')
                    is_camera = sensor_type.startswith('CAM')
                    
                    if is_radar:
                        training_method = "5-fold CV"
                        dropout_rate = 0.35
                        voting_method = "Weighted Majority Vote"
                    elif is_camera:
                        training_method = "Standard"
                        dropout_rate = 0.3
                        voting_method = "Single feature (no voting)"
                    else:
                        training_method = "Standard"
                        dropout_rate = 0.3
                        voting_method = "Weighted Majority Vote"
                    
                    f.write(f"Sensor: {sensor_type}\n")
                    f.write(f"  Training Method: {training_method}\n")
                    f.write(f"  Dropout Rate: {dropout_rate}\n")
                    f.write(f"  Voting Method: {voting_method}\n")
                    f.write(f"  Test records: {total_records}\n")
                    f.write(f"  Predicted faulty: {faulty_predicted}\n")
                    f.write(f"  Predicted healthy: {total_records - faulty_predicted}\n")
                    f.write(f"  Faulty percentage: {summary.get('faulty_percentage', 0):.1f}%\n")
                    
                    if 'feature_thresholds' in sensor_data and sensor_data['feature_thresholds']:
                        f.write(f"  Feature Thresholds and Weights:\n")
                        for feature_name, threshold in sensor_data['feature_thresholds'].items():
                            weight = sensor_data['feature_weights'].get(feature_name, 1.0)
                            f.write(f"    - {feature_name}: Threshold={threshold}, Weight={weight}\n")
                    
                    if 'feature_statistics' in sensor_data:
                        f.write(f"  Feature statistics:\n")
                        for feature_name, stats in sensor_data['feature_statistics'].items():
                            weight = sensor_data['feature_weights'].get(feature_name, 1.0)
                            threshold = stats.get('threshold', 0.5)
                            f.write(f"    - {feature_name}: {stats['faulty_count']}/{stats['total_count']} "
                                   f"({stats['faulty_percentage']:.1f}%), "
                                   f"Threshold: {threshold}, Weight: {weight}\n")
                    
                    f.write("-"*40 + "\n")
                    
                    total_tests += total_records
                    total_faulty_predicted += faulty_predicted
            
            f.write(f"\nOverall Summary:\n")
            f.write(f"  Total test records: {total_tests}\n")
            f.write(f"  Total predicted faulty: {total_faulty_predicted}\n")
            f.write(f"  Overall faulty percentage: {(total_faulty_predicted/total_tests)*100:.1f}%\n" if total_tests > 0 else "N/A\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Notes:\n")
            f.write("- Test records have actual_faulty = None (no ground truth available)\n")
            f.write("- LiDAR: Standard training with dropout=0.3, weighted majority voting\n")
            f.write("- Radar: 5-fold cross validation with dropout=0.35, weighted majority voting\n")
            f.write("- Camera: Standard training with dropout=0.3, single feature decision (no voting)\n")
            f.write("- Camera sensors have only one feature: camera_obstacle_distance_distribution_faulty (dim=40)\n")
            f.write("- Camera: No K-fold cross validation applied\n")
            f.write("- Camera threshold: 0.5 (original setting)\n")
    
    def save_models_info(self):
        """Save models information"""
        models_info = {
            'model_type': self.model.model_type,
            'device': self.model.device,
            'sensors': list(self.processor.sensor_types),
            'features': self.processor.sensor_features,
            'feature_dims': self.processor.feature_dims,
            'liDAR_dropout_rate': 0.3,
            'radar_dropout_rate': 0.35,
            'camera_dropout_rate': 0.3,
            'loss_function': 'BCEWithLogitsLoss',
            'bidirectional': True,
            'sequence_length': 4,
            'training_epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'radar_training_method': '5-fold cross validation',
            'camera_training_method': 'Standard training (no K-fold)',
            'lidar_training_method': 'standard training',
            'voting_method': 'weighted_majority_vote for LiDAR/Radar, single_feature for Camera',
            'feature_weights': self.processor.feature_weights,
            'feature_thresholds': self.processor.feature_thresholds,
            'threshold_strategy': 'Different thresholds per feature based on sensitivity and sensor noise',
            'threshold_explanations': self.processor.threshold_explanations,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        info_filename = "models_info.json"
        info_filepath = os.path.join(self.model_path, info_filename)
        
        with open(info_filepath, 'w') as f:
            json.dump(models_info, f, indent=2)
        
        print(f"Models information saved to: {info_filepath}")


def main():
    """Main function to run the system"""
    
    json_file_path = r"C:\PATH\Vector.json"
    
    model_type = 'lstm'
    
    print("="*80)
    print("Fault Detection System for LiDAR, Radar, and Camera Data")
    print(f"Using {model_type.upper()} models with PyTorch")
    print("="*80)
    print("Configuration:")
    print("  LiDAR:")
    print("    - Training: Standard")
    print("    - Dropout: 0.3")
    print("    - Voting: Weighted majority vote")
    print("  Radar:")
    print("    - Training: 5-fold cross validation")
    print("    - Dropout: 0.35")
    print("    - Voting: Weighted majority vote")
    print("  Camera (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT):")
    print("    - Training: Standard (no K-fold)")
    print("    - Dropout: 0.3")
    print("    - Single feature: camera_obstacle_distance_distribution_faulty (dim=40)")
    print("    - Threshold: 0.5")
    print("    - No voting needed (single feature decision)")
    print("="*80)
    
    system = FaultDetectionSystem(json_file_path, model_type)
    results = system.run_detection()
    
    system.save_models_info()
    
    print("\n" + "="*80)
    print("Processing completed successfully!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    results = main()
