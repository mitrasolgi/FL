# syft_utils.py

import os
import pandas as pd
import numpy as np
import syft as sy
from time import sleep
import threading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tenseal as ts

def get_ckks_context():
    poly_modulus_degree = 8192
    # Safe settings for federated learning on small models
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,  # use 16384 if model is deep
        coeff_mod_bit_sizes=[60, 40, 40, 60]  # Higher bits, deeper circuit
    )
    context.global_scale = 2 ** 40  # This scale must match your encryption/decryption logic
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context,poly_modulus_degree

def load_data():
    """Load biometric dataset from Excel files"""
    folder = "data/behaviour_biometrics_dataset/feature_kmt_dataset/feature_kmt_xlsx"
    print(f"📂 Loading data from: {folder}")

    dfs = []
    for file in os.listdir(folder):
        if file.endswith(".xlsx"):
            try:
                df = pd.read_excel(os.path.join(folder, file))
                df["user_id"] = file.replace(".xlsx", "")
                dfs.append(df)
                print(f"   ✅ Loaded {file}: {len(df)} records")
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
                continue

    data = pd.concat(dfs, ignore_index=True)
    print(f"📊 Total loaded: {len(data)} biometric records from {len(dfs)} users")
    return data


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
    """Start Syft server"""
    name = f"Factory{port - 55000}"
    node = sy.orchestra.launch(name=name, port=port, reset=True, n_consumers=1, create_producer=True)
    client = node.login(email="info@openmined.org", password="changethis")
    client.settings.allow_guest_signup(True)

    dataset = create_syft_dataset(df)
    client.upload_dataset(dataset)

    print(f"[{name}] running at {node.url}:{node.port} with {len(df)} records.")
    return node, client


def approve_requests(client):
    """Approve requests automatically"""
    while True:
        for req in client.requests:
            if req.status.value != 2:
                req.approve(approve_nested=True)
                print(f"Approved: {req.requesting_user_verify_key}")
        sleep(5)
def evaluate_biometric_model(y_true, y_pred_conf, confidence, threshold=0.5):
    """
    y_pred_conf: Predicted confidence scores (floats)
    confidence: Same as y_pred_conf unless you use a second vector
    threshold: Threshold to convert scores to binary predictions
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = (y_pred_conf > threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "avg_confidence": np.mean(confidence),
        "confidence_std": np.std(confidence)
    }

def evaluate_model(model, client_datasets):
    # Collect all test data from all clients
    X_test_all = np.vstack([data["X_test"] for data in client_datasets.values()])
    y_test_all = np.hstack([data["y_test"] for data in client_datasets.values()])
    if hasattr(model, "predict_encrypted"):
        preds, confs = model.predict_encrypted(X_test_all, threshold=0.5)
    else:
        preds, confs = model.predict(X_test_all)



    accuracy = accuracy_score(y_test_all, preds)
    precision = precision_score(y_test_all, preds, zero_division=0)
    recall = recall_score(y_test_all, preds, zero_division=0)
    f1 = f1_score(y_test_all, preds, zero_division=0)
    avg_conf = np.mean(confs)
    conf_std = np.std(confs)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_confidence": avg_conf,
        "confidence_std": conf_std,
    }
def run_federated_training_with_syft(data, ports=None):
    """Run federated training setup with Syft for N clients (default 1)"""
    if ports is None:
        ports = [55000 + i for i in range(1)]

    print(f"🌐 Setting up federated training with {len(ports)} clients")

    users = data["user_id"].unique()
    np.random.seed(42)
    np.random.shuffle(users)
    user_splits = np.array_split(users, len(ports))
    dfs = [data[data["user_id"].isin(split)] for split in user_splits]

    servers, clients = [], []

    for i, (port, df) in enumerate(zip(ports, dfs)):
        print(f"🚀 Starting server for Client {i} on port {port}")
        try:
            server, client = start_server(port, df)
            servers.append(server)
            clients.append(client)
            threading.Thread(target=approve_requests, args=(client,), daemon=True).start()
        except Exception as e:
            print(f"❌ Failed to start server {i} on port {port}: {e}")

    sleep(5)

    client_datasets = {}

    for i, client in enumerate(clients):
        try:
            df = client.datasets.get_all()[0].assets[0].data

            y = df["label"].values
            X = df.drop(columns=["label", "user_id"], errors="ignore")
            X = X.apply(pd.to_numeric, errors="coerce").dropna().values
            y = y[:len(X)]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            client_datasets[f"Client{i}"] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "port": ports[i],
                "num_users": len(user_splits[i])
            }

            print(f"✅ Client{i}: {len(X_train)} train, {len(X_test)} test samples")

        except Exception as e:
            print(f"❌ Failed to process data for Client{i}: {e}")
            continue

    return client_datasets
