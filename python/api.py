# api.py
import asyncio
import multiprocessing

from fastapi import FastAPI, WebSocket
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
import os
import time
import torch
from python.train_model import (
    connected_clients,
    agent_lock,
    get_agent,
    set_running, do_restart,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

message_queue = asyncio.Queue()  # Create a global queue
messages = [];
shutdown_event = None
process_queue: multiprocessing.Queue = None


def set_process_queue(queue):
    global process_queue
    process_queue = queue


async def read_from_process_queue():
    global shutdown_event
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()
    while not shutdown_event.is_set():
        try:
            message = await loop.run_in_executor(None, process_queue.get)
            await message_queue.put(message)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error reading from process queue: {e}")
    print("process_queue closed")

@app.get("/")
async def get():
    print(os.getcwd())
    with open(os.path.join(os.getcwd(), "static/index.html")) as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.on_event("startup")
async def startup_event():
    global message_task, process_queue_task
    process_queue_task = asyncio.create_task(read_from_process_queue())
    message_task = asyncio.create_task(send_messages_to_clients())
    pass


@app.on_event("shutdown")
async def shutdown_event():
    global message_task, process_queue_task, shutdown_event
    print("Shutting down... queues")

    shutdown_event.set()

    process_queue.put({})

    process_queue_task.cancel()
    message_task.cancel()

    try:
        await process_queue_task
    except asyncio.CancelledError:
        print("Process queue task cancelled")

    try:
        await message_task
    except asyncio.CancelledError:
        print("Message task cancelled")

    process_queue.close()
    process_queue.join_thread()
    print("Shutting down... queues finished")



async def send_messages_to_clients():
    while not shutdown_event.is_set():
        message = await message_queue.get()
        messages.append(message)
        for client in connected_clients:
            try:
                await client.send_json({
                    'action': "step",
                    'data': message
                })
            except:
                print(f"Error sending message to client")


message_task = None
process_queue_task = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        'action': "forward",
        'data': messages
    })
    connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "start":
                set_running(True)
            elif data == "pause":
                set_running(False)
            elif data == "stop":
                set_running(False)
                for client in connected_clients:
                    await client.send_json({'action': "stopped"})

            elif data == "restart":
                do_restart()
                for client in connected_clients:
                    await client.send_json({'action': "restarted"})
            elif data == "export":
                with agent_lock:
                    get_agent().save_model("model/model.pth")
                for client in connected_clients:
                    await client.send_json({'action': "exported"})
            elif data == "import":
                with agent_lock:
                    get_agent().load_model("model/model.pth")
                for client in connected_clients:
                    await client.send_json({'action': "imported"})
            time.sleep(1)
    except Exception as e:
        print(f"Websocket closed with error: {e}")
    finally:
        connected_clients.remove(websocket)
