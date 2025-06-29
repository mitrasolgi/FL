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


##############################
# 🔑 Reproducibility
##############################
set_seed_value = 42


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
class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(SimpleLinearModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


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
# 🔐 Encryption Context
##############################
def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 60],
    )
    context.global_scale = 2**40
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
def federated_training(clients, mode="plain", context=None, epochs=20, local_epochs=2, lr=0.01):
    global_model = None
    scalers = {}

    for epoch in tqdm(range(epochs), desc=f"Global Epoch ({mode})"):
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
                global_model = SimpleLinearModel(input_size=X_tensor.shape[1])

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

            if mode == "secure_aggregation":
                encrypted_weights = encrypt_model_weights(local_model, context)
                encrypted_weights_list.append(encrypted_weights)
            else:
                local_weights.append(local_model.state_dict())

        if mode == "secure_aggregation":
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
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def predict(self, encrypted_vector):
        result = encrypted_vector.dot(self.weight.tolist())
        result += float(self.bias)
        return result


def encrypted_inference(model, context, scaler, X_test):
    y_pred = []

    X_scaled = scaler.transform(X_test)

    fc_weight = model.fc.weight.data.clone()
    fc_bias = model.fc.bias.data.clone()

    encrypted_model = EncryptedLinearModel(fc_weight[0], fc_bias[0])

    for sample in X_scaled:
        encrypted_vector = ts.ckks_vector(context, sample.tolist())
        encrypted_output = encrypted_model.predict(encrypted_vector)
        decrypted_result = encrypted_output.decrypt()[0]
        probability = 1 / (1 + np.exp(-decrypted_result))
        label = int(probability > 0.5)
        y_pred.append(label)

    return np.array(y_pred)


##############################
# 🔎 Plain Inference
##############################
def evaluate_inference(model, scaler, X_test, y_test):
    X_scaled = scaler.transform(X_test)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    logits = model(X_tensor)
    probabilities = torch.sigmoid(logits).squeeze().detach().numpy()
    y_pred = (probabilities > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return acc, prec, rec, f1


##############################
# 🚀 Main Execution
##############################
if __name__ == "__main__":
    MODES = ["plain", "secure_aggregation"]

    for MODE in MODES:
        print(f"\n==============================")
        print(f"⚙️ Running in mode: {MODE}")
        print(f"==============================")

        clients = initialize_clients()

        context = None
        if MODE == "secure_aggregation":
            context = create_context()

        model, scalers = federated_training(
            clients, mode=MODE, context=context, epochs=20, local_epochs=2, lr=0.01
        )

        torch.save(model.state_dict(), f"global_model_{MODE}.pth")
        print(f"\n💾 Model saved as global_model_{MODE}.pth")

        context_inference = create_context()

        print(f"\n📊 Inference results for {MODE}")
        for idx, client in enumerate(clients):
            print(f"\nClient {idx} name: {client.name}")
            df = client.datasets.get_all()[0].assets[0].data
            print(f"Sample data for client {idx}:")
            print(df.head())

            y = df["label"]
            X = (
                df.drop(columns=["label", "user_id"], errors="ignore")
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )
            y = y.loc[X.index]

            scaler = scalers[client.name]

            # Plain Inference
            acc_p, prec_p, rec_p, f1_p = evaluate_inference(model, scaler, X, y)

            # Encrypted Inference
            y_pred_enc = encrypted_inference(model, context_inference, scaler, X)
            acc_e = accuracy_score(y.values, y_pred_enc)
            prec_e = precision_score(y.values, y_pred_enc, zero_division=0)
            rec_e = recall_score(y.values, y_pred_enc, zero_division=0)
            f1_e = f1_score(y.values, y_pred_enc, zero_division=0)

            print(f"\n📊 Client {idx} ({client.name}):")
            print(f"  Plain Inference:")
            print(f"    Accuracy:  {acc_p:.4f}")
            print(f"    Precision: {prec_p:.4f}")
            print(f"    Recall:    {rec_p:.4f}")
            print(f"    F1 Score:  {f1_p:.4f}")
            print(f"  Encrypted Inference:")
            print(f"    Accuracy:  {acc_e:.4f}")
            print(f"    Precision: {prec_e:.4f}")
            print(f"    Recall:    {rec_e:.4f}")
            print(f"    F1 Score:  {f1_e:.4f}")
