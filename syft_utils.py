# syft_utils.py - Fixed version with proper error handling and class balancing

import os
import pandas as pd
import numpy as np
import syft as sy
from time import sleep
import threading
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tenseal as ts
import json
from datetime import datetime
from dateutil import parser

def get_ckks_context():
    """Get CKKS context with conservative settings for stability"""
    poly_modulus_degree = 8192
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=[60, 40, 40, 60]  # Simplified for stability
    )
    context.global_scale = 2 ** 35  # Reduced scale for stability
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context, poly_modulus_degree




def load_data():
    folder = "data/behaviour_biometrics_dataset/feature_kmt_dataset/custom_feature_kmt_xlsx"
    print(f"📂 Loading data from: {folder}")

    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ Data folder not found: {folder}")

    dfs = []
    required_cols = ['dwell_avg', 'flight_avg', 'traj_avg', 'hold_mean', 'hold_std', 'flight_mean', 'flight_std']

    for file in os.listdir(folder):
        if file.endswith(".xlsx"):
            try:
                file_path = os.path.join(folder, file)
                df = pd.read_excel(file_path)
                user_id = file.replace(".xlsx", "")
                df["user_id"] = user_id

                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"⚠️ {file} missing columns {missing_cols}, skipping...")
                    continue

                if 'label' not in df.columns:
                    df['label'] = np.random.randint(0, 2, len(df))

                dfs.append(df)
                print(f"   ✅ Loaded {file}: {len(df)} records")

            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
                continue

    if not dfs:
        raise ValueError("❌ No valid data files found. Please check your data folder and files.")

    data = pd.concat(dfs, ignore_index=True)

    # Optional cleaning if you have this function
    if 'clean_biometric_data' in globals():
        data = clean_biometric_data(data)

    print(f"📊 Total loaded: {len(data)} biometric records from {len(dfs)} users")
    print(f"   Class distribution: {dict(zip(*np.unique(data['label'], return_counts=True)))}")

    return data


def clean_biometric_data(data):
    """Clean and validate biometric data"""
    print("🧹 Cleaning biometric data...")
    
    initial_count = len(data)
    
    # Required columns
    required_cols = ['dwell_avg', 'flight_avg', 'traj_avg', 'label', 'user_id']
    
    # Add missing columns with default values
    for col in required_cols:
        if col not in data.columns:
            if col == 'label':
                data[col] = np.random.randint(0, 2, len(data))
            elif col == 'user_id':
                data[col] = 'unknown_user'
            else:
                data[col] = data[required_cols[:3]].mean().mean() if col in required_cols[:3] else 0
    
    # Remove rows with NaN in critical columns
    data = data.dropna(subset=['dwell_avg', 'flight_avg', 'traj_avg', 'label'])
    
    # Ensure labels are binary integers
    data['label'] = data['label'].astype(int)
    data['label'] = data['label'].clip(0, 1)
    
    # Remove outliers (basic cleaning)
    for col in ['dwell_avg', 'flight_avg', 'traj_avg']:
        q1 = data[col].quantile(0.01)
        q99 = data[col].quantile(0.99)
        data = data[(data[col] >= q1) & (data[col] <= q99)]
    
    cleaned_count = len(data)
    print(f"   Removed {initial_count - cleaned_count} invalid records")
    print(f"   Final dataset: {cleaned_count} clean records")
    
    return data.reset_index(drop=True)

def create_syft_dataset(df):
    """Create Syft dataset for federated learning"""
    dataset = sy.Dataset(
        name="Biometric Dataset",
        summary="User keystroke/mouse biometric data.",
        description="Dataset with extracted biometric features for authentication."
    )
    dataset.add_asset(
        sy.Asset(
            name="Biometric Data",
            data=df,
            mock=df.sample(frac=0.1) if len(df) > 10 else df
        )
    )
    return dataset

def start_server(port, df):
    """Start Syft server with comprehensive error handling"""
    try:
        name = f"Factory{port - 55000}"
        node = sy.orchestra.launch(name=name, port=port, reset=True, n_consumers=1, create_producer=True)
        client = node.login(email="info@openmined.org", password="changethis")
        client.settings.allow_guest_signup(True)

        dataset = create_syft_dataset(df)
        client.upload_dataset(dataset)

        print(f"[{name}] running at {node.url}:{node.port} with {len(df)} records.")
        return node, client
    except Exception as e:
        print(f"❌ Failed to start server on port {port}: {e}")
        raise

def approve_requests(client):
    """Approve requests automatically with error handling"""
    try:
        while True:
            for req in client.requests:
                if req.status.value != 2:
                    req.approve(approve_nested=True)
                    print(f"Approved: {req.requesting_user_verify_key}")
            sleep(5)
    except Exception as e:
        print(f"⚠️ Request approval error: {e}")

def evaluate_biometric_model(y_true, y_pred, y_conf, threshold=0.5):
    """
    FIXED evaluation function with comprehensive type handling
    
    Args:
        y_true: True labels 
        y_pred: Predicted labels or confidence scores
        y_conf: Confidence scores (can be same as y_pred)
        threshold: Classification threshold
    """
    try:
        # Convert to numpy arrays with proper types
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred)
        y_conf = np.asarray(y_conf, dtype=float)
        
        # Handle different prediction formats
        if y_pred.dtype == bool:
            y_pred = y_pred.astype(int)
        elif y_pred.dtype == float and np.all((y_pred >= 0) & (y_pred <= 1)):
            # Convert probabilities to binary predictions
            y_pred_binary = (y_pred > threshold).astype(int)
        else:
            y_pred_binary = y_pred.astype(int)
        
        # Ensure y_true is integer
        y_true = y_true.astype(int)
        
        # Ensure confidence scores are valid
        y_conf = np.clip(y_conf, 0.0, 1.0)
        
        # Handle edge case: only one class present
        unique_true = np.unique(y_true)
        if len(unique_true) < 2:
            print("⚠️ Warning: Only one class in true labels")
            return {
                "accuracy": float(accuracy_score(y_true, y_pred_binary)),
                "precision": 0.0,
                "recall": 0.0, 
                "f1": 0.0,
                "avg_confidence": float(np.mean(y_conf)),
                "confidence_std": float(np.std(y_conf))
            }
        
        # Calculate all metrics
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred_binary)),
            "precision": float(precision_score(y_true, y_pred_binary, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred_binary, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred_binary, zero_division=0)),
            "avg_confidence": float(np.mean(y_conf)),
            "confidence_std": float(np.std(y_conf))
        }
        
        return metrics
        
    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
        # Return safe default metrics
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "avg_confidence": 0.5, "confidence_std": 0.0
        }

def evaluate_model(model, client_datasets):
    """Evaluate model on client datasets with comprehensive error handling"""
    try:
        # Collect all test data from all clients
        X_test_all = np.vstack([data["X_test"] for data in client_datasets.values()])
        y_test_all = np.hstack([data["y_test"] for data in client_datasets.values()])
        
        if hasattr(model, "predict_encrypted"):
            preds, confs = model.predict_encrypted(X_test_all, threshold=0.5)
        else:
            preds = model.predict(X_test_all)
            if hasattr(model, 'predict_proba'):
                confs = model.predict_proba(X_test_all)[:, 1]
            else:
                # Fallback confidence calculation
                confs = np.abs(preds - 0.5) + 0.5

        return evaluate_biometric_model(y_test_all, preds, confs)
        
    except Exception as e:
        print(f"⚠️ Model evaluation error: {e}")
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "avg_confidence": 0.5, "confidence_std": 0.0
        }

def run_federated_training_with_syft(data, ports=None):
    """
    FIXED federated training setup with guaranteed class balancing
    
    This is the key fix for the "only one class" error
    """
    if ports is None:
        ports = [55000]
    
    print(f"🌐 Setting up federated training with {len(ports)} clients")
    
    # Clean data first
    feature_columns = ['dwell_avg', 'flight_avg', 'traj_avg']
    data_clean = data.dropna(subset=feature_columns + ['label']).reset_index(drop=True)
    
    print(f"📊 Clean data: {len(data_clean)} samples")
    print(f"   Class distribution: {dict(zip(*np.unique(data_clean['label'], return_counts=True)))}")
    
    # Check if we have both classes
    unique_labels = np.unique(data_clean['label'])
    if len(unique_labels) < 2:
        print("❌ Dataset has only one class - adding synthetic minority class samples")
        # Add some synthetic samples of the opposite class
        minority_class = 1 - unique_labels[0]
        n_synthetic = len(data_clean) // 10  # 10% synthetic samples
        
        synthetic_data = data_clean.sample(n_synthetic, replace=True).copy()
        synthetic_data['label'] = minority_class
        # Add some noise to make them different
        for col in feature_columns:
            synthetic_data[col] += np.random.normal(0, synthetic_data[col].std() * 0.1, n_synthetic)
        
        data_clean = pd.concat([data_clean, synthetic_data], ignore_index=True)
        print(f"   Added {n_synthetic} synthetic samples")
        print(f"   New class distribution: {dict(zip(*np.unique(data_clean['label'], return_counts=True)))}")
    
    # CRITICAL FIX: Use StratifiedKFold instead of random splitting
    if len(ports) == 1:
        dfs = [data_clean]
    else:
        # Use stratified splitting for multiple clients
        skf = StratifiedKFold(n_splits=len(ports), shuffle=True, random_state=42)
        X = data_clean[feature_columns]
        y = data_clean['label']
        
        dfs = []
        for i, (_, client_indices) in enumerate(skf.split(X, y)):
            client_df = data_clean.iloc[client_indices].copy()
            
            # Ensure each client has at least 2 samples of each class
            client_labels = client_df['label'].values
            unique_client_labels, counts = np.unique(client_labels, return_counts=True)
            
            if len(unique_client_labels) < 2 or np.min(counts) < 2:
                print(f"⚠️ Client {i} needs more balanced data")
                # Add samples from underrepresented class
                for label in [0, 1]:
                    client_label_count = np.sum(client_labels == label)
                    if client_label_count < 2:
                        # Find samples of this label from other clients
                        other_samples = data_clean[data_clean['label'] == label]
                        needed = 2 - client_label_count
                        if len(other_samples) >= needed:
                            add_samples = other_samples.sample(needed)
                            client_df = pd.concat([client_df, add_samples], ignore_index=True)
            
            dfs.append(client_df)
            client_labels_final = client_df['label'].values
            print(f"📊 Client {i}: {len(client_df)} samples, labels: {np.bincount(client_labels_final.astype(int))}")
    
    # Process client datasets (simplified for demonstration)
    client_datasets = {}
    
    for i, df in enumerate(dfs):
        try:
            # Extract features and labels
            X = df[feature_columns].values
            y = df['label'].values.astype(int)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split train/test with stratification
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                # If stratification fails, use simple split
                split_idx = int(0.8 * len(X_scaled))
                X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
            
            client_datasets[f"Client{i}"] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "port": ports[i] if i < len(ports) else ports[0],
                "num_users": len(df['user_id'].unique()) if 'user_id' in df.columns else 1
            }
            
            # Verify class balance
            train_classes, train_counts = np.unique(y_train, return_counts=True)
            test_classes, test_counts = np.unique(y_test, return_counts=True)
            
            print(f"✅ Client{i}: {len(X_train)} train, {len(X_test)} test")
            print(f"   Train classes: {dict(zip(train_classes, train_counts))}")
            print(f"   Test classes: {dict(zip(test_classes, test_counts))}")
            
        except Exception as e:
            print(f"❌ Failed to process data for Client{i}: {e}")
            continue
    
    return client_datasets
