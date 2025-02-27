# sumo.py
import os
import sys

import libsumo as traci
import traci._trafficlight as tl
from sumolib import checkBinary  # noqa
import traci.constants as tc

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

SUMO_BINARY = checkBinary('sumo-gui')
SUMO_CONFIG = "sumo/osm.sumocfg"  # Path to your SUMO configuration file


# --- Action Space and Mapping ---
class TrafficLightAction:
    def __init__(self, tls_id, phase_index):
        self.tls_id = tls_id
        self.phase_index = phase_index


class SUMO:
    TLS_IDS = [
        "cluster_1766144540_276084995",  # Friedrich Garten
        "GS_cluster_1774030167_2282510415",  # Friedrich Friedhof
        "joinedS_5484739155_cluster_1766173441_33010768",  # Sutt Friedrich
    ]

    junctions = [
        "cluster_1774030167_2282510415", # Friedrich Friedhof
        "3870008290",
        "2669030870",
        "cluster_1766144540_276084995", # Friedrich Garten
        "cluster_1766173441_33010768", # Sutt Friedrich
        "5484739152"

    ]

    # --- SUMO Interaction Functions ---
    def start_sumo(self, begin_time=0):
        traci.start([SUMO_BINARY, "-b", str(begin_time), "-c", SUMO_CONFIG])  # Start SUMO temporarily

    def simulation_step(self):
        traci.simulationStep()

    def get_tls_logics(self) -> dict[str, tl.Logic]:
        return {
            tls_id: traci.trafficlight.getAllProgramLogics(tls_id)[0]
            for tls_id in self.TLS_IDS
        }

    def set_phase(self, tls_id, phase_index):
        traci.trafficlight.setProgram(tls_id, '0')
        traci.trafficlight.setPhase(tls_id, phase_index)
        traci.trafficlight.setPhaseDuration(tls_id, 9999)

    def set_rgb_phase(self, tls_id, rgb_state):
        traci.trafficlight.setRedYellowGreenState(tls_id, rgb_state)

    def get_lane_data(self, lanes: list[str]) -> dict[str, dict[int, int]]:
        return {
            lane: traci.lane.getSubscriptionResults(lane)
            for lane in lanes
        }

    def subscribe_lane_data(self, lanes: list[str]):
        for lane in lanes:
            traci.lane.subscribe(lane, (tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_MEAN_SPEED, tc.LAST_STEP_OCCUPANCY))

    def get_induction_loop_ids(self, ):
        """Retrieves the IDs of all induction loops in the simulation."""
        induction_loop_ids = traci.inductionloop.getIDList()
        return induction_loop_ids

    def get_controlled_lanes(self):
        edges = []
        for junction_id in self.junctions:
            edges.extend(traci.junction.getIncomingEdges(junction_id))

        lanes = []
        for edge in edges:
            lane_count = traci.edge.getLaneNumber(edge)
            for lane_index in range(lane_count):
                lane_id = f"{edge}_{lane_index}"
                lanes.append(lane_id)

        return lanes

    def get_last_step_vehicle_ids(self, lane: str) -> list[str]:
        return traci.lane.getLastStepVehicleIDs(lane)

    def get_vehicle_speed(self, vehicle: str) -> int:
        return traci.vehicle.getSpeed(vehicle)

    def get_vehicle_waiting_time(self, vehicle: str) -> int:
        return traci.vehicle.getWaitingTime(vehicle)

    def get_vehicle_lane(self, veh_id: str) -> str:
        return traci.vehicle.getLaneID(veh_id)

    def get_all_lanes(self) -> list[str]:
        return traci.lane.getIDList()

    def get_lane_queue_length(self, lane_id, vehicle_speeds):
        queue_length = 0
        speed_threshold = 5  # m/s

        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        for veh_id in vehicle_ids:
            if vehicle_speeds[veh_id] < speed_threshold:
                queue_length += 1

        return queue_length

    def get_total_actions(self, ):
        return sum([len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases) * 2 for tls_id in self.TLS_IDS])

    def get_all_vehicles(self, ) -> list[str]:
        return traci.vehicle.getIDList()

    def stop_sumo(self, ):
        traci.close()
