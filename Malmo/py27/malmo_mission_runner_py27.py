"""
Malmo mission runner for Python 2.7 Docker containers only. Exact copy of mission_runner
but compatible with this python version

Usage (inside Malmo container):
    cd /path/to/tests
    python py27/malmo_mission_runner_py27.py

Usage (on host):
    python model.py
"""
from __future__ import print_function

from malmo import MalmoPython
import os
import re
import signal
import socket
import sys
import time
import json
import math
import re

from lewm_integration_layer_py27 import (
    get_LeWM_action,
    process_LeWM_action,
    prepare_video_data,
)

# load selected environment from config
DEFAULT_ENV_NAME = "single_tree_navigation"
ENV_NAME = os.environ.get("LEWM_ENV", DEFAULT_ENV_NAME)
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "environments", ENV_NAME + ".xml")

with open(ENV_PATH) as env_file:
    missionXML = env_file.read()

print("Loaded environment '{0}' from {1}".format(ENV_NAME, ENV_PATH))

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print("ERROR:", e)
    print(agent_host.getUsage())
    sys.exit(1)

if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    sys.exit(0)

# Start socket server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", 25565))
server.listen(1)

# wait for client connection
print("LeWM socket listening on 0.0.0.0:25565")
connection = None

# Helpers for connecting to the socket
def exit_procedure():
    if connection is not None:
        connection.close()
    server.close()

def signal_handler(sig, frame):
    exit_procedure()
    sys.exit(0)

# Start handler for signals that come from the operating system
signal.signal(signal.SIGINT, signal_handler)

def start_mission():
    my_mission = MalmoPython.MissionSpec(missionXML, True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_mission_record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                return None
            time.sleep(2)

    print("Waiting for the mission to start ", end=" ")
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print()
    print("Mission running")
    return world_state


def stop_mission():
    try:
        world_state = agent_host.getWorldState()
        if world_state.is_mission_running:
            agent_host.sendCommand("quit")
            deadline = time.time() + 5
            while world_state.is_mission_running and time.time() < deadline:
                time.sleep(0.1)
                world_state = agent_host.getWorldState()
    except RuntimeError as e:
        print("Error stopping mission:", e)


# Begin mission
episode = 0
try:
    while True:
        episode += 1
        print("=== Mission {0} ===".format(episode))
        
        world_state = start_mission()
        if world_state is None:
            break

        print("Listening for connection...")
        connection, address = server.accept()
        print("Client connected from", address)

        setting = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        try:
            while world_state.is_mission_running:
                world_state = agent_host.getWorldState()
                action_list = []

                for error in world_state.errors:
                    print("Error:", error.text)

                if world_state.number_of_video_frames_since_last_state > 0:
                    # get physics stats from latest observation
                    # view full observations list here: https://github.com/trueagi-io/Vereya/blob/master/api.md
                    physics = {"x": None, "y": None, "z": None, "yaw": None, "pitch": None}
                    if world_state.number_of_observations_since_last_state > 0:
                        raw_stats = json.loads(world_state.observations[-1].text)
                        physics = {
                            "x": raw_stats.get("XPos"),
                            "y": raw_stats.get("YPos"),
                            "z": raw_stats.get("ZPos"),
                            "yaw": raw_stats.get("Yaw"),
                            "pitch": raw_stats.get("Pitch"),
                        }
                    print(physics)

                    # Send frame and physics to client and wait for action.
                    frame = prepare_video_data(world_state.video_frames[-1])
                    action = get_LeWM_action(frame, physics, connection, setting)
                    action_list = process_LeWM_action(action, setting)
                    print(action_list)

                for action in action_list:
                    agent_host.sendCommand(action)

                time.sleep(0.05)
        except socket.error as e:
            print("Client disconnected:", e)
        finally:
            if connection is not None:
                connection.close()
                connection = None
            stop_mission()

        print("Mission ended")
finally:
    exit_procedure()
