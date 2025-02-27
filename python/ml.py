# ml.py

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sortedcontainers import SortedList
import threading

from python.sumo import TrafficLightAction


# ------------------------------
# 1) Dueling DQN architecture
# ------------------------------
class DuelingDQN(nn.Module):
    """
    Dueling network splits into (Value) and (Advantage) streams.
    We'll still produce multiple heads (one per traffic light).
    """

    def __init__(self, input_size, output_sizes, num_tls, TLS_IDS):
        super(DuelingDQN, self).__init__()
        self.num_tls = num_tls
        self.TLS_IDS = TLS_IDS
        self.output_sizes = output_sizes

        hidden_size = 128

        # Common feature extractor
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # For each traffic light, we have a separate dueling head:
        # Value stream (scalar) + Advantage stream (one per action).
        self.value_layers = nn.ModuleList()
        self.advantage_layers = nn.ModuleList()
        for size in output_sizes:
            self.value_layers.append(nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ))
            self.advantage_layers.append(nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, size)
            ))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Returns a list of Q-value tensors, one for each traffic light.
        """
        # Shared feature
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        outputs = []
        for value_layer, advantage_layer in zip(self.value_layers, self.advantage_layers):
            value = value_layer(x)  # shape: (batch_size, 1)
            advantage = advantage_layer(x)  # shape: (batch_size, num_actions)

            # Dueling combination
            # Q = V + (A - mean(A))
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
            outputs.append(q)
        return outputs

    def get_actions(self, q_values):
        """
        Given a list of Q-value tensors (batch_size=1 in inference),
        pick the best action for each traffic light.
        """
        actions = []
        for i, q_values_tls in enumerate(q_values):
            action_index = torch.argmax(q_values_tls).item()
            phase_index = action_index
            actions.append(TrafficLightAction(self.TLS_IDS[i], phase_index))
        return actions


# ------------
# ReplayMemory
# ------------
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --------------------------
# Optionally: PrioritizedReplayMemory (unused by default)
# --------------------------
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = SortedList()
        self.alpha = alpha
        self.lock = threading.Lock()

    def push(self, *experience):
        max_priority = max(self.priorities) if self.priorities else 1.0
        with self.lock:
            self.memory.append(experience)
            self.priorities.add(max_priority ** self.alpha)
            if len(self.priorities) > self.capacity:
                self.priorities.pop(0)

    def sample(self, batch_size, beta=0.4):
        with self.lock:
            if len(self.memory) == self.capacity:
                priorities = np.array(self.priorities)
                probabilities = priorities / priorities.sum()
            else:
                probabilities = np.ones(len(self.memory)) / len(self.memory)

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        with self.lock:
            for idx, td_error in zip(indices, td_errors):
                priority = np.mean(td_error)
                del self.priorities[idx]
                self.priorities.add(float(priority) ** self.alpha)

    def __len__(self):
        return len(self.memory)


class RLAgent:
    def __init__(
            self,
            input_size,
            num_tls,
            logics,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.95,
            batch_size=128,
            replay_memory_size=50000,
            tau=0.01,  # Polyak update factor
    ):
        self.output_sizes = [len(logics[logic].phases) for logic in logics]
        self.input_size = input_size
        TLS_IDS = list(logics.keys())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use dueling architecture
        self.model = DuelingDQN(input_size, self.output_sizes, num_tls, TLS_IDS).to(self.device)
        self.target_model = DuelingDQN(input_size, self.output_sizes, num_tls, TLS_IDS).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=30, verbose=True, min_lr=1e-7)

        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = ReplayMemory(replay_memory_size)

        # Use Huber loss for stability
        self.loss_fn = nn.SmoothL1Loss()
        self.loss_history = []

        self.logics = logics
        self.num_tls = num_tls

        # Soft update factor
        self.tau = tau


    def reset_episode(self):
        """
        decaying epsilon each episode
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        pass

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.random() > self.epsilon:
            with torch.no_grad():
                # shape of state is (input_size,); add batch dimension
                q_values = self.model(state.unsqueeze(0))
                return self.model.get_actions(q_values)
        else:
            actions = []
            for tls_id in self.logics.keys():
                logics = self.logics[tls_id]
                num_phases = len(logics.phases)
                phase_index = random.randrange(0, num_phases)
                actions.append(TrafficLightAction(tls_id, phase_index))
            return actions

    def train(self, state, action, reward, next_state, done, step):
        """
        Standard DQN training step with Double DQN target calculation.
        """
        self.memory.push(state, [a.phase_index for a in action], reward, next_state, done)

        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.stack(state_batch).to(self.device)
        next_state_batch = torch.stack(next_state_batch).to(self.device)
        action_batch = torch.tensor(action_batch, device=self.device)  # shape: (batch_size, num_tls)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)  # shape: (batch_size,)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device)  # shape: (batch_size,)

        # ----------------
        # Double DQN logic
        # ----------------
        with torch.no_grad():
            # Online network to select best actions
            next_q_online = self.model(next_state_batch)  # list of shape [ (batch_size, #actions_i), ... ]
            # Argmax for each traffic light
            best_next_actions = [
                torch.argmax(q_vals, dim=1)  # shape: (batch_size,)
                for q_vals in next_q_online
            ]

            # Target network to evaluate
            next_q_target = self.target_model(next_state_batch)

            # For each traffic light i, gather the Q-value of the best_next_actions[i]
            max_next_q_values_per_tls = []
            for i, q_vals_target in enumerate(next_q_target):
                # q_vals_target shape: (batch_size, #actions_i)
                chosen_actions = best_next_actions[i].unsqueeze(1)  # (batch_size, 1)
                max_q = q_vals_target.gather(1, chosen_actions).squeeze(1)  # (batch_size,)
                max_next_q_values_per_tls.append(max_q)

            # Combine across all traffic lights by summation or average
            # sum them so each time step's total Q is the sum across traffic lights
            max_next_q_values = torch.stack(max_next_q_values_per_tls, dim=1).sum(dim=1)  # shape: (batch_size,)

            # Target = R + gamma * max_next_q_values
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values

        # --------------------------
        # Current Q-values (online)
        # --------------------------
        current_q_list = self.model(state_batch)  # list of Q-values, one per TLS
        # For each TLS i, gather the Q-value of the chosen action in action_batch[:, i]
        # Then sum across all TLS to get a single scalar Q per sample
        chosen_q_values_per_tls = []
        for i, q_vals in enumerate(current_q_list):
            chosen_actions_i = action_batch[:, i].unsqueeze(1)  # shape: (batch_size,1)
            chosen_q_i = q_vals.gather(1, chosen_actions_i).squeeze(1)  # shape: (batch_size,)
            chosen_q_values_per_tls.append(chosen_q_i)

        current_q_sum = torch.stack(chosen_q_values_per_tls, dim=1).sum(dim=1)  # shape: (batch_size,)

        # ------------
        # Compute loss
        # ------------
        loss = self.loss_fn(current_q_sum, target_q_values)
        self.loss_history.append(loss.item())

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # -----------
        # Soft update
        # -----------
        for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'epsilon_decay': self.epsilon_decay,
            'memory': [
                (s.cpu(), a, r, ns.cpu(), d)
                for (s, a, r, ns, d) in self.memory.memory
            ],
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        if not os.path.isfile(filepath):
            print(f"No model found at {filepath}")
            return

        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.target_model.to(self.device)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.loss_history = checkpoint['loss_history']
        self.epsilon_decay = checkpoint['epsilon_decay']

        self.memory.memory = deque(
            [(state.to(self.device), action, reward, next_state.to(self.device), done)
             for state, action, reward, next_state, done in checkpoint['memory']],
            maxlen=self.memory.capacity
        )

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        print(f"Model loaded from {filepath}")
