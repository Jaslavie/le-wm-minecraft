"""
Python 2.7 integration layer for Malmo Docker containers.

Use this module ONLY inside Malmo images that ship Python 2.7 + MalmoPython.
Pair with test_malmo_mission_runner_py27.py on the container and model.py (Python 3.10+)
on the host over localhost:25565.
"""
from __future__ import print_function

import select
import pickle
import struct

def get_LeWM_action(frame, stats, connection, setting):
    # [ forward, left, back, right, jump, sneak, sprint, attack, camera, camera ]
    connection.sendall(frame)
    stats_bytes = pickle.dumps(stats, protocol=2)
    connection.sendall(struct.pack(">I", len(stats_bytes)))
    connection.sendall(stats_bytes)

    data = b""
    while not data:
        ready, _, _ = select.select([connection], [], [])
        if not ready:
            return setting
        chunk = connection.recv(4096)
        if not chunk:
            return setting
        data += chunk

    action = pickle.loads(data)
    if action == "":
        action = setting
    return action


def process_LeWM_action(action_list, current):
    instructions = []

    move = action_list[0] - action_list[2]
    strafe = action_list[3] - action_list[1]

    if move != current[0] - current[2]:
        instructions.append("move {0}".format(move))

    # Strafe disabled for fixed-view single-tree navigation.
    # if strafe != current[3] - current[1]:
    #     instructions.append("strafe {0}".format(strafe))

    # Jump/sneak/sprint are disabled for fixed-view navigation.
    # if current[4] != action_list[4]:
    #     instructions.append("jump {0}".format(action_list[4]))
    #
    # if current[5] != action_list[5]:
    #     instructions.append("crouch {0}".format(action_list[5]))

    # Re-issue attack every tick while attacking so Malmo sustains continuous mining;
    # sending it once did not accumulate block-break progress. Release (0) only on change.
    if action_list[7] != 0 or current[7] != action_list[7]:
        instructions.append("attack {0}".format(int(action_list[7])))

    # Camera disabled for fixed-view navigation ablation.
    # if action_list[8] != current[8]:
    #     instructions.append("turn {0}".format(action_list[8]))
    #
    # if action_list[9] != current[9]:
    #     instructions.append("pitch {0}".format(action_list[9]))

    for i, action in enumerate(action_list):
        current[i] = action

    return instructions


def prepare_video_data(frame_data):
    pixels = frame_data.pixels
    if isinstance(pixels, (bytes, bytearray)):
        return bytes(pixels)
    return bytes(bytearray(pixels))
