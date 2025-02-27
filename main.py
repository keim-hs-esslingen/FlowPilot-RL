# main.py
import asyncio
import multiprocessing
import signal
import sys
import threading
import time

import uvicorn

from python.api import app, set_process_queue
from python.train_model import run_simulation, trigger_shutdown
from python.sumo import SUMO
from python.traffic_generation import TrafficGen

stop_event = threading.Event()

process_queue = multiprocessing.Queue()


def run_sumo_simulation():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def stop_simulation():
        """Gracefully shut down the server when the stop event is set."""
        while not stop_event.is_set():
            await asyncio.sleep(1)
        trigger_shutdown()

    loop.create_task(stop_simulation())
    loop.run_until_complete(run_simulation(SUMO(), process_queue))
    loop.close()


def start_api():
    """Start the FastAPI server in a separate thread."""
    set_process_queue(process_queue)
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, workers=1)
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def stop_server():
        """Gracefully shut down the server when the stop event is set."""
        while not stop_event.is_set():
            await asyncio.sleep(1)
        await server.shutdown()
        loop.stop()

    loop.create_task(stop_server())
    loop.create_task(server.serve())
    loop.run_forever()


def signal_handler(sig, frame):
    """Handle Ctrl+C signal to stop the program."""
    print("Stopping the program...")
    stop_event.set()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    api_thread = threading.Thread(target=start_api)
    sim_thread = threading.Thread(target=run_sumo_simulation)
    sim_thread.start()
    api_thread.start()

    while True:
        try:
            time.sleep(1)
            if stop_event.is_set():
                break
        except KeyboardInterrupt:
            print("KeyboardInterrupt received.")
            break

    print("Waiting for threads to finish...")
    sim_thread.join()
    api_thread.join()
    print("Program stopped.")
    sys.exit(0)