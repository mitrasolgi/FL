import multiprocessing
import logging
import time
from threading import Event, Thread
import server


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def start_server_process(sid, client_data):
    logging.info(f"Starting server {sid}...")
    datasite, client = server.spawn_server(sid, client_data)
    if datasite and client:
        logging.info(f"Server {sid} started.")
        stop_event = Event()
        approval_thread = Thread(target=server.check_and_approve_incoming_requests, args=(client, stop_event))
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
    num_servers = 3

    # Load and partition data once
    full_data = server.load_full_data()
    data_partitions = server.partition_data(full_data, num_servers)

    processes = []

    for sid in range(num_servers):
        client_data = data_partitions[sid]
        p = multiprocessing.Process(target=start_server_process, args=(sid, client_data))
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