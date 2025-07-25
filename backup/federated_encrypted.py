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
import syft as sy
import tenseal as ts

##############################
# 🔧 CONFIGURATION
##############################
MODE = 'secure_aggregation'  # 'plain' or 'secure_aggregation'
set_seed_value = 42
MODEL_TYPE = 'MLP'  # 'MLP' or 'CNN'

##############################
# 🔑 Reproducibility
##############################
def set_seed(seed=set_seed_value):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

##############################
# 🧠 Model
##############################
class MLPModel(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def get_intermediate_output(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return x
class CNNModel(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(CNNModel, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(16, output_size)

    def forward(self, x):
        x = x.view(-1, 1, self.input_size)  # Reshape for Conv1d: (batch, channel, features)
        x = self.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)  # Shape: (batch, channels)
        x = self.dropout(x)
        return self.fc(x)

    def get_intermediate_output(self, x):
        x = x.view(-1, 1, self.input_size)
        x = self.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return x

##############################
# 🔗 Client Setup
##############################
def initialize_clients():
    return [
        sy.login(email="info@openmined.org", password="changethis", port=55000),
        sy.login(email="info@openmined.org", password="changethis", port=55001),
        sy.login(email="info@openmined.org", password="changethis", port=55002),
    ]

##############################
# 📊 Metrics
##############################
def calculate_metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
    )

##############################
# 🔐 Encryption Context
##############################
def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context

##############################
# 🔐 Secure Aggregation Helpers
##############################
def encrypt_model_weights(model, context):
    encrypted_weights = {}
    for key, param in model.state_dict().items():
        array = param.cpu().numpy().flatten().tolist()
        encrypted_weights[key] = ts.ckks_vector(context, array)
    return encrypted_weights

def aggregate_encrypted_weights(encrypted_list):
    aggregated = {}
    keys = encrypted_list[0].keys()
    for key in keys:
        summed = encrypted_list[0][key]
        for enc in encrypted_list[1:]:
            summed += enc[key]
        aggregated[key] = summed
    return aggregated

def decrypt_and_load_model(aggregated, model, num_clients):
    state_dict = model.state_dict()
    for key in state_dict.keys():
        decrypted = np.array(aggregated[key].decrypt()) / num_clients
        reshaped = decrypted.reshape(state_dict[key].shape)
        state_dict[key] = torch.tensor(reshaped, dtype=state_dict[key].dtype)
    model.load_state_dict(state_dict)

##############################
# 🔀 Federated Training
##############################
def federated_training(clients, context=None, epochs=20, local_epochs=3, lr=0.001):
    global_model = None
    scalers = {}

    for epoch in tqdm(range(epochs), desc="Global Epoch"):
        local_weights = []
        encrypted_weights_list = []

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

            X_train, _, y_train, _ = train_test_split(
                X_scaled, y, test_size=0.2, random_state=set_seed_value, stratify=y
            )

            X_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

            if global_model is None:
                if MODEL_TYPE == 'MLP':
                    global_model = MLPModel(input_size=X_tensor.shape[1])
                elif MODEL_TYPE == 'CNN':
                    global_model = CNNModel(input_size=X_tensor.shape[1])
                else:
                    raise ValueError("Unknown MODEL_TYPE")


            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=lr)
            loss_fn = nn.BCEWithLogitsLoss()

            local_model.train()
            for _ in range(local_epochs):
                optimizer.zero_grad()
                output = local_model(X_tensor)
                loss = loss_fn(output, y_tensor)
                loss.backward()
                optimizer.step()

            if MODE == 'secure_aggregation':
                encrypted_weights = encrypt_model_weights(local_model, context)
                encrypted_weights_list.append(encrypted_weights)
            else:
                local_weights.append(local_model.state_dict())

        if MODE == 'secure_aggregation':
            aggregated = aggregate_encrypted_weights(encrypted_weights_list)
            decrypt_and_load_model(aggregated, global_model, len(clients))
        else:
            new_weights = {}
            for key in local_weights[0].keys():
                stacked = torch.stack([w[key].float() for w in local_weights])
                new_weights[key] = torch.mean(stacked, dim=0)
            global_model.load_state_dict(new_weights)

    return global_model, scalers

##############################
# 🔐 Encrypted Inference
##############################
class EncryptedLinearModel:
    def __init__(self, weight, bias, context):
        self.weight = weight
        self.bias = bias
        self.context = context

    def predict(self, encrypted_vector):
        result = []
        for i in range(self.weight.shape[0]):
            w_row = self.weight[i].tolist()
            enc_dot = encrypted_vector.dot(w_row)
            enc_dot = enc_dot + float(self.bias[i].item())
            result.append(enc_dot)
        return result[0]

def encrypted_inference(model, context, scaler, X_test):
    y_pred = []

    X_scaled = scaler.transform(X_test)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    intermediate_X = model.get_intermediate_output(X_tensor).detach().numpy()

    if hasattr(model, 'fc2'):
        fc_weights = model.fc2.weight.data.clone()
        fc_bias = model.fc2.bias.data.clone()
    elif hasattr(model, 'fc'):
        fc_weights = model.fc.weight.data.clone()
        fc_bias = model.fc.bias.data.clone()
    else:
        raise AttributeError("Model has no final fully connected layer named 'fc2' or 'fc'")

    encrypted_model = EncryptedLinearModel(fc_weights, fc_bias, context)

    for sample in intermediate_X:
        encrypted_vector = ts.ckks_vector(context, sample.tolist())
        encrypted_output = encrypted_model.predict(encrypted_vector)
        decrypted_result = encrypted_output.decrypt()[0]
        probability = 1 / (1 + np.exp(-decrypted_result))
        label = int(probability > 0.5)
        y_pred.append(label)

    return np.array(y_pred)

##############################
# 🔎 Plain inference helper
##############################
def evaluate_inference(model, scaler, X_test, y_test):
    X_scaled = scaler.transform(X_test)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    logits = model(X_tensor)
    probabilities = torch.sigmoid(logits).squeeze().detach().numpy()
    y_pred = (probabilities > 0.5).astype(int)

    acc, prec, rec, f1 = calculate_metrics(y_test.values, y_pred)
    return acc, prec, rec, f1

##############################
# 🚀 Main
##############################
if __name__ == "__main__":
    clients = initialize_clients()

    context = None
    if MODE == 'secure_aggregation':
        context = create_context()

    print(f"⚙️ Running in mode: {MODE}")

    model, scalers = federated_training(clients, context)

    torch.save(model.state_dict(), f"global_model_{MODE}.pth")
    print(f"\n💾 Model saved as global_model_{MODE}.pth")

    print(f"\n📊 Inference comparison (plain vs encrypted)")
    context_inference = create_context()

    for idx, client in enumerate(clients):
        df = client.datasets.get_all()[0].assets[0].data
        y = df["label"]
        X = df.drop(columns=["label", "user_id"], errors="ignore").apply(pd.to_numeric, errors="coerce").dropna()
        y = y.loc[X.index]

        scaler = scalers[client.name]

        acc_p, prec_p, rec_p, f1_p = evaluate_inference(model, scaler, X, y)

        y_pred_enc = encrypted_inference(model, context_inference, scaler, X)
        acc_e, prec_e, rec_e, f1_e = calculate_metrics(y.values, y_pred_enc)

        print(f"\nClient {idx}:")
        print(f"  Plain Inference:")
        print(f"    Accuracy:  {acc_p:.4f}")
        print(f"    Precision: {prec_p:.4f}")
        print(f"    Recall:    {rec_p:.4f}")
        print(f"    F1:        {f1_p:.4f}")
        print(f"  Encrypted Inference:")
        print(f"    Accuracy:  {acc_e:.4f}")
        print(f"    Precision: {prec_e:.4f}")
        print(f"    Recall:    {rec_e:.4f}")
        print(f"    F1:        {f1_e:.4f}")
