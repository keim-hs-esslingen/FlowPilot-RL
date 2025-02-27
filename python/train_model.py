# train_model.py

import asyncio
import random
import threading
import multiprocessing
from collections import deque
from python.ml import RLAgent
from python.simulation import extract_features, calculate_reward, TrafficLightState, reset_smoothed_reward
from python.test_model import test_model

# --- Global variables ---
shutdown = False
connected_clients = []
agent_lock = threading.Lock()
agent = None
running = False
restart = False


def get_agent() -> RLAgent:
    return agent


def set_running(is_running):
    global running
    running = is_running


def do_restart():
    global restart
    restart = True


def trigger_shutdown():
    global shutdown
    shutdown = True


async def run_simulation(sumo, process_queue):
    from python.api import message_queue
    global running, restart, reward, agent, shutdown, smoothed_reward

    sumo.start_sumo(random.randint(40000, 80000))
    time_window = 1
    features_per_loop = 3
    history_length = 10

    lanes = sumo.get_controlled_lanes()
    input_size_features = len(lanes) * features_per_loop * time_window

    logics = sumo.get_tls_logics()
    total_phases = sum(len(logic.phases) for logic in logics.values())
    input_size_tls = total_phases * history_length + len(sumo.TLS_IDS) * history_length

    agent = RLAgent(input_size_features + input_size_tls, num_tls=len(sumo.TLS_IDS), logics=logics)

    running = False
    step = 0
    epoch = 0

    advance_steps = random.randint(100, 200)
    for _ in range(advance_steps):
        sumo.simulation_step()

    tls_state = {tls_id: TrafficLightState(0, -10, len(logics[tls_id].phases)) for tls_id in sumo.TLS_IDS}
    for tls_id in sumo.TLS_IDS:
        sumo.set_phase(tls_id, 0)
    sumo.subscribe_lane_data(lanes)

    for tls_id in tls_state:
        tls_state[tls_id].phase_history = deque([[0] * len(logics[tls_id].phases)] * history_length, maxlen=history_length)
        tls_state[tls_id].yellow_history = deque([False] * history_length, maxlen=history_length)

    while not shutdown:
        if restart:
            # Start a new process for testing
            test_process = multiprocessing.Process(target=test_model, args=(
                agent.input_size, agent.num_tls, agent.model.state_dict(), epoch, process_queue))
            test_process.start()

            sumo.stop_sumo()
            sumo.start_sumo(random.randint(40000, 80000))
            advance_steps = random.randint(100, 200)
            for _ in range(advance_steps):
                sumo.simulation_step()
            sumo.subscribe_lane_data(lanes)
            tls_state = {tls_id: TrafficLightState(0, -10, len(logics[tls_id].phases)) for tls_id in sumo.TLS_IDS}
            for tls_id in sumo.TLS_IDS:
                sumo.set_phase(tls_id, 0)
            agent.reset_episode()
            reset_smoothed_reward()
            epoch += 1
            step = 0
            restart = False

            for tls_id in tls_state:
                tls_state[tls_id].phase_history = deque([[0] * len(logics[tls_id].phases)] * history_length, maxlen=history_length)
                tls_state[tls_id].yellow_history = deque([False] * history_length, maxlen=history_length)

        if running:

            lane_data = sumo.get_lane_data(lanes)
            state = extract_features(lane_data, tls_state, lane_ids=lanes, history_length=history_length)

            with agent_lock:
                state = state.to_tensor()
                action = agent.select_action(state)

            for tls in action:
                tls_id = tls.tls_id
                new_phase = tls.phase_index
                curr_tls_state = tls_state[tls_id]
                if curr_tls_state.phase != new_phase and (step - curr_tls_state.step) >= 3:
                    if not curr_tls_state.yellow:
                        curr_tls_state.yellow = True
                        curr_tls_state.step = step
                        yellow_phase = logics[tls_id].phases[curr_tls_state.phase].state.replace("G", "y").replace("g", "y")
                        sumo.set_rgb_phase(tls_id, yellow_phase)
                    elif (step - curr_tls_state.step) >= 3:
                        curr_tls_state.yellow = False
                        curr_tls_state.step = step
                        curr_tls_state.phase = new_phase
                        sumo.set_phase(tls_id, new_phase)

                curr_tls_state.phase_history.append(curr_tls_state.get_one_hot_phase())
                curr_tls_state.yellow_history.append(curr_tls_state.yellow)

            sumo.simulation_step()
            step += 1

            next_lane_data = sumo.get_lane_data(lanes)
            next_state = extract_features(next_lane_data, tls_state, lane_ids=lanes, history_length=history_length)
            reward = calculate_reward(sumo)

            done = reward < -1 or step > 5000

            with agent_lock:
                next_state = next_state.to_tensor()
                agent.train(state, action, reward, next_state, done, step)

            if step % 5 == 0:
                await asyncio.sleep(0.001)
                await message_queue.put(
                    {'step': step, 'reward': reward, 'epoch': epoch, 'loss': agent.loss_history[-1] if agent.loss_history else 0})

            if step % 500 == 0:
                await message_queue.put(
                    {'avg_loss': sum(agent.loss_history[-500:]) / 500})

            if done:
                do_restart()

        else:
            await asyncio.sleep(0.1)
    sumo.stop_sumo()
    print("sumo stopped")
