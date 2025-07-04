import multiprocessing
import logging
import time
from threading import Event, Thread
import server
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def start_server_process(sid, client_data, port):
    logging.info(f"Starting server {sid} on port {port}...")

    datasite, client = server.spawn_server(client_data, sid, port)

    if datasite and client:
        logging.info(f"Server {sid} started on port {port}.")
        stop_event = Event()
        approval_thread = Thread(target=server.check_and_approve_requests, args=(client,))
        approval_thread.start()

        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            logging.info(f"Server {sid} stopping.")
            stop_event.set()
            approval_thread.join()
    else:
        logging.error(f"Server {sid} failed to start.")


if __name__ == "__main__":
    full_data = server.load_full_data()

    # 🔥 Split data by user_id
    users = full_data["user_id"].unique()
    base_port = 55000

    processes = []

    for i, user in enumerate(users):
        client_data = full_data[full_data["user_id"] == user]
        port = base_port + i

        p = multiprocessing.Process(target=start_server_process, args=(i, client_data, port))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logging.info("Terminating all servers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        logging.info("All servers terminated.")
