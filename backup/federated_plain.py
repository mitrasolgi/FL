import os
import copy
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import syft as sy

# Set global random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class CNNModel(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(CNNModel, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(2)
        x = self.dropout(x)
        return self.fc(x)

def initialize_clients():
    domain_1 = sy.login(email="info@openmined.org", password="changethis", port=55000)
    domain_2 = sy.login(email="info@openmined.org", password="changethis", port=55001)
    domain_3 = sy.login(email="info@openmined.org", password="changethis", port=55002)
    return [domain_1, domain_2, domain_3]

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    return accuracy, precision, recall, f1

def aggregate_metrics(client_metrics):
    metrics_to_aggregate = ["accuracy", "precision", "recall", "f1"]
    aggregated = {}
    for metric in metrics_to_aggregate:
        last_values = []
        for client in client_metrics:
            values = client_metrics[client].get(metric, [])
            if values:
                last_values.append(values[-1])
        aggregated[metric] = np.mean(last_values) if last_values else None
    return aggregated

def federated_training(clients, epochs=20, local_epochs=2, learning_rate=0.001, patience=3):
    global_model = None
    client_metrics = {}
    best_loss = float('inf')
    early_stopping_counter = 0
    client_scalers = {}

    for epoch in tqdm(range(epochs), desc="Global Epochs"):
        local_weights = []
        epoch_loss = 0

        for client in tqdm(clients, desc="Clients", leave=False):
            datasets = client.datasets.get_all()
            if not datasets:
                continue

            df = datasets[0].assets[0].data
            y = df["label"]
            df = df.drop(columns=["label", "user_id"], errors="ignore")
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            y = y.loc[df.index]

            if df.empty or y.empty:
                continue

            # Use one consistent scaler per client
            if client.name not in client_scalers:
                scaler = StandardScaler()
                df_scaled = scaler.fit_transform(df)
                client_scalers[client.name] = scaler
            else:
                df_scaled = client_scalers[client.name].transform(df)

            X_train, X_test, y_train, y_test = train_test_split(
                df_scaled, y, test_size=0.2, random_state=42
            )

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

            if global_model is None:
                global_model = CNNModel(input_size=X_train_tensor.shape[1])

            model = copy.deepcopy(global_model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            loss_fn = nn.BCEWithLogitsLoss()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

            model.train()
            for _ in range(local_epochs):
                optimizer.zero_grad()
                output = model(X_train_tensor)
                loss = loss_fn(output, y_train_tensor)
                loss.backward()
                optimizer.step()
                scheduler.step()

            local_weights.append(copy.deepcopy(model.state_dict()))

            client_name = client.name
            if client_name not in client_metrics:
                client_metrics[client_name] = {"accuracy": [], "precision": [], "recall": [], "f1": []}

            model.eval()
            with torch.no_grad():
                output = model(X_test_tensor)
                predicted_labels = (output > 0.5).long().squeeze()
                y_true = y_test_tensor.squeeze().long()
                accuracy, precision, recall, f1 = calculate_metrics(
                    y_true.detach().numpy(), predicted_labels.detach().numpy())

            client_metrics[client_name]["accuracy"].append(accuracy)
            client_metrics[client_name]["precision"].append(precision)
            client_metrics[client_name]["recall"].append(recall)
            client_metrics[client_name]["f1"].append(f1)

            epoch_loss += loss.item()

        if local_weights:
            new_state_dict = copy.deepcopy(local_weights[0])
            for key in new_state_dict:
                new_state_dict[key] = new_state_dict[key].float()
                for i in range(1, len(local_weights)):
                    new_state_dict[key] += local_weights[i][key].float()
                new_state_dict[key] /= len(local_weights)
            global_model.load_state_dict(new_state_dict)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    overall_metrics = aggregate_metrics(client_metrics)

    print("\nPer-client metrics (last epoch):")
    for client_name, metrics in client_metrics.items():
        print(f"{client_name}:")
        print(f"  Accuracy:  {metrics['accuracy'][-1]}")
        print(f"  Precision: {metrics['precision'][-1]}")
        print(f"  Recall:    {metrics['recall'][-1]}")
        print(f"  F1 Score:  {metrics['f1'][-1]}")

    print("\nOverall aggregated metrics (averaged across clients):")
    print(f"Accuracy:  {overall_metrics['accuracy']}")
    print(f"Precision: {overall_metrics['precision']}")
    print(f"Recall:    {overall_metrics['recall']}")
    print(f"F1 Score:  {overall_metrics['f1']}")

    return global_model

if __name__ == "__main__":
    clients = initialize_clients()
    model = federated_training(clients)
    if model:
        torch.save(model.state_dict(), "global_model_plain.pth")
        print("Model saved: global_model_plain.pth")
