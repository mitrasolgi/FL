import os
import pandas as pd
import numpy as np
import syft as sy
from time import sleep
from collections import Counter


TOTAL_ROUNDS = 5
PORT = 55000


def load_full_data():
    folder_path = "data/behaviour_biometrics_dataset/feature_kmt_dataset/feature_kmt_xlsx"
    data_frames = []
    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            full_path = os.path.join(folder_path, file)
            df = pd.read_excel(full_path, engine='openpyxl')
            df["user_id"] = os.path.splitext(file)[0]
            data_frames.append(df)

    if not data_frames:
        raise RuntimeError(f"No Excel files found in {folder_path}.")

    data = pd.concat(data_frames, ignore_index=True)
    return data


def create_syft_dataset(client_data: pd.DataFrame) -> sy.Dataset:
    if "label" in client_data.columns:
        class_distribution = dict(Counter(client_data["label"]))
        print(f"Class distribution: {class_distribution}")
    else:
        print("No 'label' column found to compute class distribution.")

    dataset = sy.Dataset(
        name="Behavioral Biometrics Dataset",
        summary="Keystroke and Mouse Tracking features from users.",
        description="Dataset containing feature-extracted biometric data collected using KMT tool."
    )

    dataset.add_asset(
        sy.Asset(name="KMT Biometric Data", data=client_data, mock=client_data.sample(frac=0.1, random_state=42))
    )
    return dataset


def spawn_server(client_data: pd.DataFrame, sid: int, port: int):
    name = f"Factory{sid}"

    data_site = sy.orchestra.launch(name=name, port=port, reset=True, n_consumers=1, create_producer=True)
    client = data_site.login(email="info@openmined.org", password="changethis")
    client.settings.allow_guest_signup(True)

    dataset = create_syft_dataset(client_data)
    client.upload_dataset(dataset)
    print(f"[{name}] Dataset uploaded with {len(client_data)} rows.")
    print(f"[{name}] Datasite running at {data_site.url}:{data_site.port}")

    return data_site, client

def check_and_approve_requests(client):
    print("✅ Approval started")

    round_num = 0
    while round_num < TOTAL_ROUNDS:
        requests = client.requests
        selected_ids = []

        for req in filter(lambda r: r.status.value != 2, requests):
            selected_ids.append(str(req.requesting_user_verify_key))
            req.approve(approve_nested=True)

        if selected_ids:
            print(f"[Round {round_num}] Approved: {selected_ids}")
            round_num += 1

        sleep(5)


if __name__ == "__main__":
    full_data = load_full_data()

    print("\n=== Dataset summary ===")
    unique_users = full_data["user_id"].nunique()
    total_samples = len(full_data)
    print(f"Client: {unique_users} unique users, {total_samples} total samples")
    print("========================\n")

    data_site, client = spawn_server(full_data)

    try:
        check_and_approve_requests(client)
    except KeyboardInterrupt:
        print("Shutting down server...")
        data_site.stop()
        print("Server stopped.")
