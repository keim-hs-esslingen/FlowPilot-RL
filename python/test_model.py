# test_model.py
import asyncio
from collections import deque

import torch

from python.ml import RLAgent
import random

from python.simulation import extract_features, calculate_reward, TrafficLightState


def test_model(input_size, num_tls, model_state_dict, epoch, queue):
    from python.sumo import SUMO
    sumo = SUMO()
    sumo.start_sumo(0)

    logics = sumo.get_tls_logics()
    lanes = sumo.get_controlled_lanes()

    test_agent = RLAgent(input_size, num_tls, logics, epsilon_start=1.0)
    test_agent.model.load_state_dict(model_state_dict)

    tls_state = {tls_id: TrafficLightState(0, -10, len(logics[tls_id].phases)) for tls_id in sumo.TLS_IDS}
    for tls_id in sumo.TLS_IDS:
        sumo.set_phase(tls_id, 0)
    sumo.subscribe_lane_data(lanes)

    for tls_id in tls_state:
        tls_state[tls_id].phase_history = deque([[0] * len(logics[tls_id].phases)] * 10, maxlen=10)
        tls_state[tls_id].yellow_history = deque([False] * 10, maxlen=10)

    step = 0
    done = False
    rewards = []

    while not done:
        lane_data = sumo.get_lane_data(lanes)
        state = extract_features(lane_data, tls_state, lane_ids=lanes, history_length=10)
        state = state.to_tensor()

        with torch.no_grad():
            action = test_agent.select_action(state)

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

        reward = calculate_reward(sumo)
        rewards.append(reward)

        if step == 10:
            test_agent.epsilon = 0

        if reward < -1 or step > 2000:
            done = True

    while len(rewards) < 2000:
        rewards.append(-1)

    avg_reward = sum(rewards) / len(rewards)
    queue.put({'test_reward': avg_reward, 'epoch': epoch})
    print(f"Test completed. Average reward: {avg_reward}")
    sumo.stop_sumo()

