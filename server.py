import os
import pandas as pd
import numpy as np
import syft as sy
from threading import Thread, Event
from time import sleep
from collections import Counter

NUM_CLIENTS = 3
TOTAL_ROUNDS = 5

DATASITE_PORTS = {f"Factory{i}": 55000 + i for i in range(NUM_CLIENTS)}
DATASITE_URLS = {name: f"http://localhost:{port}" for name, port in DATASITE_PORTS.items()}


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


def partition_data(data, num_partitions):
    unique_users = data["user_id"].unique()
    user_partitions = np.array_split(unique_users, num_partitions)
    partitions = []
    for users in user_partitions:
        part_df = data[data["user_id"].isin(users)].reset_index(drop=True)
        partitions.append(part_df)
    return partitions


def print_partition_info(data_partitions):
    for i, part in enumerate(data_partitions):
        unique_users = part["user_id"].nunique()
        total_samples = len(part)
        print(f"Client {i}: {unique_users} unique users, {total_samples} total samples")
        users = part["user_id"].unique()
        print(f"Client {i} users: {users}\n")


def create_syft_dataset_for_client(name: str, client_data: pd.DataFrame) -> sy.Dataset:
    if "label" in client_data.columns:
        class_distribution = dict(Counter(client_data["label"]))
        print(f"[{name}] Class distribution: {class_distribution}")
    else:
        print(f"[{name}] No 'label' column found to compute class distribution.")

    dataset = sy.Dataset(
        name="Behavioral Biometrics Dataset",
        summary="Keystroke and Mouse Tracking features from 88 users.",
        description="Dataset containing feature-extracted biometric data collected using KMT tool."
    )

    dataset.add_asset(
        sy.Asset(name="KMT Biometric Data", data=client_data, mock=client_data.sample(frac=0.1, random_state=42))
    )
    return dataset


def spawn_server(sid: int, client_data: pd.DataFrame):
    name = f"Factory{sid}"
    port = DATASITE_PORTS[name]

    data_site = sy.orchestra.launch(name=name, port=port, reset=True, n_consumers=NUM_CLIENTS, create_producer=True)
    client = data_site.login(email="info@openmined.org", password="changethis")
    client.settings.allow_guest_signup(True)

    ds = create_syft_dataset_for_client(name, client_data)
    client.upload_dataset(ds)
    print(f"[{name}] Dataset uploaded with {len(client_data)} rows.")

    print(f"[{name}] Datasite running at {data_site.url}:{data_site.port}")
    return data_site, client


def check_and_approve_incoming_requests(client, stop_event: Event):
    print(f"✅ Approval thread started for {client.name}")

    round_num = 0
    while not stop_event.is_set() and round_num < TOTAL_ROUNDS:
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
    data_partitions = partition_data(full_data, NUM_CLIENTS)

    print("\n=== Dataset partition summary ===")
    print_partition_info(data_partitions)
    print("================================\n")

    stop_events = []
    threads = []

    for sid in range(NUM_CLIENTS):
        stop_event = Event()
        stop_events.append(stop_event)

        def server_thread(sid=sid, stop_event=stop_event):
            client_data = data_partitions[sid]
            data_site, client = spawn_server(sid, client_data)
            check_and_approve_incoming_requests(client, stop_event)

        t = Thread(target=server_thread)
        t.start()
        threads.append(t)

    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        print("Shutting down servers...")
        for stop_event in stop_events:
            stop_event.set()
        for t in threads:
            t.join()
        print("All servers stopped.")
