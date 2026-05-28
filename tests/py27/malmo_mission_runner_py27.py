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
import random
import signal
import socket
import sys
import time

from lewm_integration_layer_py27 import (
    get_LeWM_action,
    process_LeWM_action,
    prepare_video_data,
)

def draw_tree(x, y, z, r):
    gen = ""

    leaf_br = r
    leaf_base_y = y + 2
    leaf_base_height = 2

    leaf_ur = 1
    leaf_upper_y = leaf_base_y + leaf_base_height
    leaf_upper_height = 2

    gen += (
        '<DrawCuboid x1="{0}" y1="{1}" z1="{2}" '
        'x2="{3}" y2="{4}" z2="{5}" type="leaves"/>\n'
    ).format(
        x - leaf_br, leaf_base_y, z - leaf_br,
        x + leaf_br, leaf_base_y + leaf_base_height - 1, z + leaf_br,
    )
    gen += (
        '<DrawCuboid x1="{0}" y1="{1}" z1="{2}" '
        'x2="{3}" y2="{4}" z2="{5}" type="leaves"/>\n'
    ).format(
        x - leaf_ur, leaf_upper_y, z - leaf_ur,
        x + leaf_ur, leaf_upper_y + leaf_upper_height - 1, z + leaf_ur,
    )

    for i in range(leaf_base_y, leaf_base_y + leaf_base_height):
        gen += '<DrawBlock x="{0}" y="{1}" z="{2}" type="air"/>\n'.format(x + leaf_br, i, z + leaf_br)
        gen += '<DrawBlock x="{0}" y="{1}" z="{2}" type="air"/>\n'.format(x + leaf_br, i, z - leaf_br)
        gen += '<DrawBlock x="{0}" y="{1}" z="{2}" type="air"/>\n'.format(x - leaf_br, i, z + leaf_br)
        gen += '<DrawBlock x="{0}" y="{1}" z="{2}" type="air"/>\n'.format(x - leaf_br, i, z - leaf_br)

    for i in range(leaf_upper_y + 1, leaf_upper_y + leaf_upper_height):
        gen += '<DrawBlock x="{0}" y="{1}" z="{2}" type="air"/>\n'.format(x + leaf_ur, i, z + leaf_ur)
        gen += '<DrawBlock x="{0}" y="{1}" z="{2}" type="air"/>\n'.format(x + leaf_ur, i, z - leaf_ur)
        gen += '<DrawBlock x="{0}" y="{1}" z="{2}" type="air"/>\n'.format(x - leaf_ur, i, z + leaf_ur)
        gen += '<DrawBlock x="{0}" y="{1}" z="{2}" type="air"/>\n'.format(x - leaf_ur, i, z - leaf_ur)

    trunk_top = leaf_upper_y + leaf_upper_height / 2
    for dy in range(int(y), int(trunk_top)):
        gen += '<DrawBlock x="{0}" y="{1}" z="{2}" type="log"/>\n'.format(x, dy, z)

    return gen


def draw_trees(num_trees, xmin, xmax, zmin, zmax, ground_y=4, use_random=True, border=False):
    gen = ""
    tree_radius = 2

    gen += (
        '<DrawCuboid x1="{0}" y1="{1}" z1="{2}" '
        'x2="{3}" y2="{4}" z2="{5}" type="air"/>\n'
    ).format(xmin - 10, ground_y, zmin - 10, xmax + 10, ground_y + 30, zmax + 10)
    gen += (
        '<DrawCuboid x1="{0}" y1="{1}" z1="{2}" '
        'x2="{3}" y2="{4}" z2="{5}" type="grass"/>\n'
    ).format(xmin - 10, ground_y - 1, zmin - 10, xmax + 10, ground_y - 1, zmax + 10)

    if border:
        gen += (
            '<DrawCuboid x1="{0}" y1="{1}" z1="{2}" '
            'x2="{3}" y2="{4}" z2="{5}" type="barrier"/>\n'
        ).format(
            xmin - tree_radius - 1, ground_y, zmin - tree_radius - 1,
            xmax + tree_radius + 1, ground_y + 4, zmax + tree_radius + 1,
        )
        gen += (
            '<DrawCuboid x1="{0}" y1="{1}" z1="{2}" '
            'x2="{3}" y2="{4}" z2="{5}" type="air"/>\n'
        ).format(
            xmin - tree_radius, ground_y, zmin - tree_radius,
            xmax + tree_radius, ground_y + 4, zmax + tree_radius,
        )

    if use_random:
        for _ in range(num_trees):
            x = random.randint(xmin, xmax)
            z = random.randint(zmin, zmax)
            gen += draw_tree(x, ground_y, z, tree_radius)
    else:
        trees_placed = 0
        step = -1 * (2 * tree_radius + 1 + 1)
        for x in range(xmax - 1, xmin, step):
            for z in range(zmax - 1, zmin, step):
                if trees_placed < num_trees:
                    gen += draw_tree(x, ground_y, z, tree_radius)
                    trees_placed += 1

    return gen


tree_xml = draw_trees(1, -5, 5, -5, 5, use_random=False, border=True)

missionXML = """<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

  <About>
    <Summary>Superflat world with random trees</Summary>
  </About>

  <ServerSection>
    <ServerInitialConditions>
      <Time>
        <StartTime>1000</StartTime>
        <AllowPassageOfTime>false</AllowPassageOfTime>
      </Time>
      <Weather>clear</Weather>
    </ServerInitialConditions>

    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,2*3,2;1;"/>
      <DrawingDecorator>
        {0}
      </DrawingDecorator>
      <ServerQuitFromTimeUp timeLimitMs="60000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>

  <AgentSection mode="Survival">
    <Name>TreeBot</Name>
    <AgentStart>
      <Placement x="0.5" y="5" z="0.5" yaw="0"/>
    </AgentStart>
    <AgentHandlers>
      <VideoProducer want_depth="false">
            <Width>{1}</Width>
            <Height>{2}</Height>
        </VideoProducer>
      <ObservationFromFullStats/>
      <ContinuousMovementCommands turnSpeedDegs="180"/>
    </AgentHandlers>
  </AgentSection>

</Mission>
""".format(tree_xml, 64, 64)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 25565))
server.listen(1)

print("Listening for connection...")
connection, address = server.accept()


def exit_procedure():
    connection.close()
    server.close()


def signal_handler(sig, frame):
    exit_procedure()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

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
            sys.exit(1)
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
print("Mission running ")

setting = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

while world_state.is_mission_running:
    world_state = agent_host.getWorldState()
    action_list = []

    for error in world_state.errors:
        print("Error:", error.text)

    if world_state.number_of_video_frames_since_last_state > 0:
        frame = prepare_video_data(world_state.video_frames[-1])
        action = get_LeWM_action(frame, connection, setting)
        action_list = process_LeWM_action(action, setting)
        print(action_list)

    for action in action_list:
        agent_host.sendCommand(action)

    time.sleep(0.1)

exit_procedure()
print()
print("Mission ended")
