# simulation.py

import traci.constants as tc
from sklearn.preprocessing import MinMaxScaler
import torch

min_occupancy = 0
max_occupancy = 1
min_count = 0
max_count = 15
min_speed = 0
max_speed = 50
reward = 0

scaler = MinMaxScaler()
scaler.fit([[min_occupancy, min_count, min_speed],
            [max_occupancy, max_count, max_speed]])


class TrafficLightState:
    def __init__(self, phase: int, step: int, num_phases: int):
        self.phase = phase
        self.step = step
        self.num_phases = num_phases
        self.yellow = False

    def get_one_hot_phase(self):
        one_hot = [0] * self.num_phases
        one_hot[self.phase] = 1
        return one_hot


class TrafficState:
    maxCount = 0
    maxSpeed = 0

    def __init__(self, occupancy, count, average_speed, tls_phase_history, tls_yellow_history):
        self.occupancy = occupancy
        self.count = count
        self.average_speed = average_speed
        self.tls_phase_history = tls_phase_history
        self.tls_yellow_history = tls_yellow_history

    def flatten(self, xss):
        return [x for xs in xss for x in xs]

    def to_tensor(self):
        combined_features = list(zip(self.occupancy, self.count, self.average_speed))
        transformed_features = scaler.transform(combined_features)
        features_tensor = torch.tensor(transformed_features, dtype=torch.float32).flatten().to('cuda')

        tls_phase_tensor = torch.tensor(self.flatten(self.tls_phase_history), dtype=torch.float32).flatten().to('cuda')
        tls_yellow_tensor = torch.tensor(self.tls_yellow_history, dtype=torch.float32).flatten().to('cuda')
        return torch.cat([features_tensor, tls_phase_tensor, tls_yellow_tensor])



reward_smoothing_factor = 0.99
smoothed_reward = 0.8


def extract_features(lane_data, tls_state, time_window=1, lane_ids=None, history_length=10):
    occupancy, count, average_speed, tls_phase_history, tls_yellow_history = [], [], [], [], []
    for lane_id in lane_ids:
        for _ in range(time_window):
            occupancy.append(lane_data[lane_id][tc.LAST_STEP_OCCUPANCY])
            count.append(lane_data[lane_id][tc.LAST_STEP_VEHICLE_NUMBER])
            average_speed.append(lane_data[lane_id][tc.LAST_STEP_MEAN_SPEED])
    for tls_id in tls_state:
        tls_phase_history.append([item for sublist in tls_state[tls_id].phase_history for item in sublist])
        tls_yellow_history.append(list(tls_state[tls_id].yellow_history))
    return TrafficState(occupancy, count, average_speed, tls_phase_history, tls_yellow_history)


def reset_smoothed_reward():
    global smoothed_reward
    smoothed_reward = 0.8

def calculate_reward(sumo):
    global smoothed_reward

    all_vehicles = sumo.get_all_vehicles()
    all_lanes = sumo.get_all_lanes()

    total_halting_vehicles = 0
    total_waiting_time = 0
    vehicle_speeds = {}
    for veh_id in all_vehicles:
        vehicle_speed = sumo.get_vehicle_speed(veh_id)
        vehicle_speeds[veh_id] = vehicle_speed
        if vehicle_speed < 0.1:
            total_halting_vehicles += 1
            total_waiting_time += sumo.get_vehicle_waiting_time(veh_id)

    total_queue_penalty = 0
    for lane_id in all_lanes:
        lane_queue_length = sumo.get_lane_queue_length(lane_id, vehicle_speeds)
        total_queue_penalty += max(0, lane_queue_length - 3)

    reward = (-total_halting_vehicles - 0.2 * total_waiting_time - 40 * total_queue_penalty) * 0.1


    reward = (reward + 200) / 200
    smoothed_reward = reward_smoothing_factor * smoothed_reward + (1 - reward_smoothing_factor) * reward

    return smoothed_reward