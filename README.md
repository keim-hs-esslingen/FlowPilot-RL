# FlowPilot-RL
FlowPilot-RL is a modular framework for adaptive traffic light control that combines microscopic traffic simulation with reinforcement learning. This repository provides a scalable environment to experiment with AI-driven traffic management strategies, integrating real-time data, simulation tools, and a user-friendly interface.

## Overview
Urban traffic congestion leads to longer travel times, higher fuel consumption, and increased emissions. Traditional traffic control systems—often based on fixed or actuated signal timings—struggle to respond to dynamic conditions like time-of-day fluctuations, weather, and incidents. FlowPilot-RL addresses these challenges by:

 - **Integrating SUMO Simulation**: Leverage the Simulation of Urban Mobility (SUMO) for realistic, microscopic traffic modeling.
 - **Employing Reinforcement Learning**: Utilize AI (implemented in PyTorch) to dynamically adjust signal timings based on real-time traffic states.
 - **Providing Real-Time Monitoring**: Use a WebAPI and interactive WebUI for live simulation control and training visualization.

## Features
- **Modular Architectur**e: The system is divided into key components:
    - WebAPI: Facilitates user interaction and real-time monitoring.
    - Simulation Module: Coordinates traffic simulation and AI agent communication.
    - SUMO Interface: Manages interactions with the SUMO simulator.
    - AI Agent: Implements reinforcement learning models using PyTorch.
- **Realistic Data Integration**: Incorporates real-world traffic data via InfluxDB pipelines.
- **Flexible Experimentation**: Easily swap out components (e.g., simulation engine or ML framework) to adapt to different scenarios.
- **User Interface**: An interactive web dashboard for starting/stopping simulations and viewing key performance metrics (e.g., rewards, losses).

## System Architecture
The framework’s architecture is designed to ensure a loosely coupled system, promoting flexibility and ease of component replacement. Key components include:

- **Simulation Coordination**: The simulation.py module retrieves traffic states via the sumo.py interface and communicates with the AI agent (ml.py).
- **Traffic Light Control**: A state machine enforces safe transitions between traffic light states.
- **Real-Time Visualization**: The Flask-based backend and WebSocket-enabled frontend provide live updates of simulation metrics.

## Getting Started
### Prerequisites
- Python 3.7+
- SUMO: For microscopic traffic simulation.
- PyTorch: For neural network and reinforcement learning implementation.

### Installation
1. Clone the Repository:

```bash
git clone https://github.com/keim-hs-esslingen/FlowPilot-RL.git
cd FlowPilot-RL
```
2. Install Dependencies:

``` bash
pip install -r requirements.txt
```

### Running the Simulation
Start the simulation by running:


```bash
python main.py
```
Then, open your browser and navigate to http://localhost:5000 to access the WebUI for real-time monitoring and control.

## Evaluation
The framework has been evaluated using a branching Deep Q-Network (DQN) with an epsilon-greedy exploration strategy, experience replay, and Double Q-Learning to address overestimation bias. Initial results from training on a realistic urban network (modeled on the city center of Aalen) demonstrate a clear improvement in traffic flow and congestion reduction.

## Future Work
Future enhancements may include:

- **Additional Data Sources**: Integrating pedestrian dynamics and weather data for improved realism.
- **Advanced RL Algorithms**: Experimenting with hybrid models and alternative exploration strategies.
- **Network-Level Control**: Scaling the framework for large-scale, network-wide traffic management.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

