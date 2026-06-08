from __future__ import print_function
from builtins import range
import sys
import os

# FIX: Ensure the current directory is in sys.path to find lewm_integration_layer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lewm_integration_layer import get_LeWM_action, process_LeWM_action, prepare_video_data

import MalmoPython
import time
import signal
import socket

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
else:
    import functools
    print = functools.partial(print, flush=True)

def draw_tree(x, y, z, r):
    gen = ""
    leaf_br = r
    leaf_base_y = y + 2
    leaf_base_height = 2
    leaf_ur = 1
    leaf_upper_y = leaf_base_y + leaf_base_height
    leaf_upper_height = 2

    gen += (f'<DrawBlock x="{x}" y="{leaf_base_y}" z="{z}" type="log"/>\n')
    gen += (f'<DrawBlock x="{x}" y="{leaf_upper_y}" z="{z}" type="log"/>\n')

    for i in range(leaf_base_y, leaf_base_y+leaf_base_height):
        gen += f'<DrawBlock x="{x+1}" y="{i}" z="{z}" type="leaves"/>\n'
        gen += f'<DrawBlock x="{x-1}" y="{i}" z="{z}" type="leaves"/>\n'
        gen += f'<DrawBlock x="{x}" y="{i}" z="{z+1}" type="leaves"/>\n'
        gen += f'<DrawBlock x="{x}" y="{i}" z="{z-1}" type="leaves"/>\n'

    for i in range(leaf_upper_y+1, leaf_upper_y+leaf_upper_height):
        gen += f'<DrawBlock x="{x+1}" y="{i}" z="{z}" type="leaves"/>\n'
        gen += f'<DrawBlock x="{x-1}" y="{i}" z="{z}" type="leaves"/>\n'
        gen += f'<DrawBlock x="{x}" y="{i}" z="{z+1}" type="leaves"/>\n'
        gen += f'<DrawBlock x="{x}" y="{i}" z="{z-1}" type="leaves"/>\n'

    for dy in range(y, leaf_upper_y + leaf_upper_height // 2):
        gen += f'<DrawBlock x="{x}" y="{dy}" z="{z}" type="log"/>\n'
    return gen

def draw_trees(num_trees, xmin, xmax, zmin, zmax, ground_y=4, random=True, border=False):
    gen = ""
    tree_radius = 2
    tree_height = 4
    gen += (f'<DrawCuboid x1="{xmin}" y1="{ground_y}" z1="{zmin}" x2="{xmax}" y2="{ground_y+tree_height}" z2="{zmax}" type="air"/>\n')
    gen += (f'<DrawCuboid x1="{xmin}" y1="{ground_y-1}" z1="{zmin}" x2="{xmax}" y2="{ground_y-1}" z2="{zmax}" type="grass"/>\n')
    
    if border:
        gen += (f'<DrawCuboid x1="{xmin}" y1="{ground_y}" z1="{zmin}" x2="{xmax}" y2="{ground_y+3}" z2="{zmin}" type="glass"/>\n')
        gen += (f'<DrawCuboid x1="{xmin}" y1="{ground_y}" z1="{zmax}" x2="{xmax}" y2="{ground_y+3}" z2="{zmax}" type="glass"/>\n')

    if random:
        import random as rand
        for _ in range(num_trees):
            x = rand.randint(xmin, xmax)
            z = rand.randint(zmin, zmax)
            gen += draw_tree(x, ground_y, z, tree_radius)
    else:
        trees_placed = 0
        for x in range(xmax-1, xmin, -1 * (2 * tree_radius + 1 + 1)):
            for z in range(zmax-1, zmin, -1 * (2 * tree_radius + 1 + 1)):
                if trees_placed < num_trees:
                    gen += draw_tree(x, ground_y, z, tree_radius)
                    trees_placed += 1
    return gen

# Environment Selection
env_type = sys.argv[1] if len(sys.argv) > 1 else "superflat"

if env_type == "forest":
    # Seed -2744534680298546054 naturally spawns a Birch/Oak forest.
    # Omitting <Placement> ensures Malmo uses the seed's natural safe spawn coordinates.
    missionXML = f'''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com">
        <About><Summary>Birch/Oak Forest VoE Test</Summary></About>
        <ModSettings><MsPerTick>1</MsPerTick></ModSettings>
        <ServerSection>
            <ServerInitialConditions>
                <Seed>-2744534680298546054</Seed>
                <Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime></Time>
            </ServerInitialConditions>
            <ServerHandlers>
                <DefaultWorldGenerator seed="-2744534680298546054"/>
                <ServerQuitFromTimeUp timeLimitMs="120000"/>
                <ServerQuitWhenAnyAgentFinishes/>
            </ServerHandlers>
        </ServerSection>
        <AgentSection mode="Survival">
            <Name>VoE_Bot</Name>
            <AgentStart>
                <Pitch>30</Pitch><Yaw>0</Yaw>
            </AgentStart>
            <AgentHandlers>
                <VideoProducer want_depth="false"><Width>64</Width><Height>64</Height></VideoProducer>
                <DiscreteMovementCommands/>
            </AgentHandlers>
        </AgentSection>
    </Mission>'''
else:
    tree_xml = draw_trees(1, -5, 5, -5, 5, random=False, border=True)
    missionXML = f'''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com">
        <About><Summary>Superflat world with single tree</Summary></About>
        <ModSettings><MsPerTick>1</MsPerTick></ModSettings>
        <ServerSection>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,2*3,2;1;clear" forceReset="true"/>
                <DrawingDecorator>{tree_xml}</DrawingDecorator>
                <ServerQuitFromTimeUp timeLimitMs="120000"/>
                <ServerQuitWhenAnyAgentFinishes/>
            </ServerHandlers>
        </ServerSection>
        <AgentSection mode="Survival">
            <Name>TreeBot</Name>
            <AgentStart>
                <Placement x="0.5" y="4.0" z="3.5" pitch="30" yaw="0"/>
            </AgentStart>
            <AgentHandlers>
                <VideoProducer want_depth="false"><Width>64</Width><Height>64</Height></VideoProducer>
                <DiscreteMovementCommands/>
            </AgentHandlers>
        </AgentSection>
    </Mission>'''

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 25565))
server.listen(1)
print(f"Listening for connection... (Env: {env_type})")
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
    agent_host.parse(sys.argv[2:]) # Pass remaining args to Malmo
except RuntimeError as e:
    print('ERROR:', e)
    exit(1)

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
            exit(1)
        else:
            time.sleep(2)

print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
print("\nMission running ")

setting = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
done = False

while world_state.is_mission_running and not done:
    world_state = agent_host.getWorldState()
    while world_state.number_of_video_frames_since_last_state == 0 and world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        
    if not world_state.is_mission_running:
        done = True
    else:
        frame = prepare_video_data(world_state.video_frames[-1])
        action = get_LeWM_action(frame, connection, setting)
        action_list = process_LeWM_action(action, setting, True)
        if action_list:
            for act in action_list:
                agent_host.sendCommand(act)
        time.sleep(0.05)

exit_procedure()
print("\nMission ended")