import os
import copy
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tenseal as ts
import syft as sy


##########################
# 🔑 Reproducibility
##########################
set_seed_value = 42

def set_seed(seed=set_seed_value):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()


##########################
# 🔗 Client Setup
##########################
def initialize_clients():
    return [
        sy.login(email="info@openmined.org", password="changethis", port=55000),
        sy.login(email="info@openmined.org", password="changethis", port=55001),
        sy.login(email="info@openmined.org", password="changethis", port=55002),
    ]


##########################
# 🔐 HE Context
##########################
def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 60],
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context


##########################
# 🧠 Model (Simple Linear)
##########################
class SimpleLinearModel:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0

    def forward(self, x):
        return np.dot(self.weights, x) + self.bias

    def forward_encrypted(self, enc_x):
        result = enc_x.dot(self.weights.tolist())
        result += self.bias
        return result

    def train_on_sample_encrypted(self, enc_x, y_true, lr):
        enc_pred = self.forward_encrypted(enc_x)
        y_vec = ts.ckks_vector(enc_x.context(), [y_true] * enc_pred.size())
        enc_error = enc_pred - y_vec
        error = enc_error.decrypt()[0]

        x_plain = np.array(enc_x.decrypt())

        self.weights -= lr * error * x_plain
        self.bias -= lr * error

    def train_on_sample_plain(self, x, y_true, lr):
        pred = self.forward(x)
        error = pred - y_true
        self.weights -= lr * error * x
        self.bias -= lr * error


##########################
# 🔐 Encrypt Model Weights
##########################
def encrypt_model_weights(model, context):
    encrypted_weights = ts.ckks_vector(context, model.weights.tolist())
    encrypted_bias = ts.ckks_vector(context, [model.bias])
    return encrypted_weights, encrypted_bias


def aggregate_encrypted_weights(weights_list, bias_list):
    sum_weights = weights_list[0]
    sum_bias = bias_list[0]

    for w, b in zip(weights_list[1:], bias_list[1:]):
        sum_weights += w
        sum_bias += b

    return sum_weights, sum_bias


def decrypt_and_load_model(enc_weights, enc_bias, model, num_clients):
    weights = np.array(enc_weights.decrypt()) / num_clients
    bias = enc_bias.decrypt()[0] / num_clients
    model.weights = weights
    model.bias = bias


##########################
# 🔀 Federated Training
##########################
def federated_training(clients, context, mode="encrypted", epochs=10, lr=0.1):
    global_model = None
    scalers = {}

    for epoch in tqdm(range(epochs), desc=f"Global Epoch ({mode.capitalize()})"):
        enc_weight_list = []
        enc_bias_list = []
        local_weights = []

        for client in clients:
            df = client.datasets.get_all()[0].assets[0].data
            y = df["label"]
            X = df.drop(columns=["label", "user_id"], errors="ignore")
            X = X.apply(pd.to_numeric, errors="coerce").dropna()
            y = y.loc[X.index]

            if client.name not in scalers:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                scalers[client.name] = scaler
            else:
                X_scaled = scalers[client.name].transform(X)

            encrypted_X = [ts.ckks_vector(context, sample.tolist()) for sample in X_scaled]
            y_values = y.values

            if global_model is None:
                global_model = SimpleLinearModel(input_size=X_scaled.shape[1])

            local_model = copy.deepcopy(global_model)

            if mode == "encrypted":
                for enc_x, y_true in zip(encrypted_X, y_values):
                    local_model.train_on_sample_encrypted(enc_x, y_true, lr)

                enc_w, enc_b = encrypt_model_weights(local_model, context)
                enc_weight_list.append(enc_w)
                enc_bias_list.append(enc_b)

            else:  # Plain training
                for x, y_true in zip(X_scaled, y_values):
                    local_model.train_on_sample_plain(x, y_true, lr)

                local_weights.append((local_model.weights, local_model.bias))

        if mode == "encrypted":
            agg_w, agg_b = aggregate_encrypted_weights(enc_weight_list, enc_bias_list)
            decrypt_and_load_model(agg_w, agg_b, global_model, len(clients))

        else:  # Aggregate plain
            stacked_w = np.stack([w for w, b in local_weights])
            stacked_b = np.array([b for w, b in local_weights])

            avg_w = np.mean(stacked_w, axis=0)
            avg_b = np.mean(stacked_b)

            global_model.weights = avg_w
            global_model.bias = avg_b

    return global_model, scalers


##########################
# 🔎 Evaluation
##########################
def evaluate_model(model, scalers, clients, title="Model"):
    all_y_true = []
    all_y_pred = []

    for client in clients:
        df = client.datasets.get_all()[0].assets[0].data
        y = df["label"]
        X = df.drop(columns=["label", "user_id"], errors="ignore")
        X = X.apply(pd.to_numeric, errors="coerce").dropna()
        y = y.loc[X.index]

        scaler = scalers[client.name]
        X_scaled = scaler.transform(X)

        for x, y_true in zip(X_scaled, y):
            pred = model.forward(x)
            label = 1 if pred > 0 else 0
            all_y_true.append(y_true)
            all_y_pred.append(label)

    acc = accuracy_score(all_y_true, all_y_pred)
    prec = precision_score(all_y_true, all_y_pred, zero_division=0)
    rec = recall_score(all_y_true, all_y_pred, zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, zero_division=0)

    print(f"\n📊 {title} Performance:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")


##########################
# 🚀 Main
##########################
if __name__ == "__main__":
    context = create_context()

    clients = initialize_clients()

    # 🔥 Plain Training
    plain_model, plain_scalers = federated_training(
        clients, context=context, mode="plain", epochs=10, lr=0.05
    )

    evaluate_model(plain_model, plain_scalers, clients, title="Plain Model")

    # 🔐 Encrypted Training
    enc_model, enc_scalers = federated_training(
        clients, context=context, mode="encrypted", epochs=10, lr=0.05
    )

    evaluate_model(enc_model, enc_scalers, clients, title="Encrypted Model")
