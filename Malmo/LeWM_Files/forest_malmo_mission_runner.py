from __future__ import print_function
from builtins import range
from lewm_integration_layer import get_LeWM_action, process_LeWM_action, prepare_video_data
import MalmoPython
import os
import sys
import time
import signal
import socket
import random

# FIX: Set a random seed so the "random" forest layout is exactly the same (fixed) every run
random.seed(42)

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

def draw_tree(x, y, z, r, tree_type="oak"):
    gen = ""
    # Base leaf config
    leaf_br = r # Leaf base radius
    leaf_base_y = y + 2
    leaf_base_height = 2
    # Upper leaf config
    leaf_ur = 1 # Leaf upper radius
    leaf_upper_y = leaf_base_y + leaf_base_height
    leaf_upper_height = 2
    
    # Malmo 1.11.2 block variants for Oak and Birch
    if tree_type == "birch":
        log_attr = 'type="log" variant="birch"'
        leaf_attr = 'type="leaves" variant="birch"'
    else:
        log_attr = 'type="log" variant="oak"'
        leaf_attr = 'type="leaves" variant="oak"'

    # Generate trunk
    for dy in range(y, leaf_upper_y + leaf_upper_height // 2):
        gen += f'<DrawBlock x="{x}" y="{dy}" z="{z}" {log_attr}/>\n'

    # Generate base leaves
    gen += f'<DrawCuboid x1="{x-r}" y1="{leaf_base_y}" z1="{z-r}" x2="{x+r}" y2="{leaf_base_y+leaf_base_height-1}" z2="{z+r}" {leaf_attr}/>\n'
    
    # Generate upper leaves
    gen += f'<DrawCuboid x1="{x-leaf_ur}" y1="{leaf_upper_y}" z1="{z-leaf_ur}" x2="{x+leaf_ur}" y2="{leaf_upper_y+leaf_upper_height-1}" z2="{z+leaf_ur}" {leaf_attr}/>\n'
    
    # Trim base leaves corners
    for i in range(leaf_base_y, leaf_base_y+leaf_base_height):
        gen += f'<DrawBlock x="{x-r}" y="{i}" z="{z-r}" type="air"/>\n'
        gen += f'<DrawBlock x="{x+r}" y="{i}" z="{z-r}" type="air"/>\n'
        gen += f'<DrawBlock x="{x-r}" y="{i}" z="{z+r}" type="air"/>\n'
        gen += f'<DrawBlock x="{x+r}" y="{i}" z="{z+r}" type="air"/>\n'
        
    # Trim upper leaves corners
    for i in range(leaf_upper_y+1, leaf_upper_y+leaf_upper_height):
        gen += f'<DrawBlock x="{x-leaf_ur}" y="{i}" z="{z-leaf_ur}" type="air"/>\n'
        gen += f'<DrawBlock x="{x+leaf_ur}" y="{i}" z="{z-leaf_ur}" type="air"/>\n'
        gen += f'<DrawBlock x="{x-leaf_ur}" y="{i}" z="{z+leaf_ur}" type="air"/>\n'
        gen += f'<DrawBlock x="{x+leaf_ur}" y="{i}" z="{z+leaf_ur}" type="air"/>\n'

    return gen

def draw_trees(num_trees, xmin, xmax, zmin, zmax, ground_y=4, random_placement=True):
    gen = ""
    tree_radius = 2
    
    # Clear original trees (SUPERFLAT ONLY)
    gen += f'<DrawCuboid x1="{xmin}" y1="{ground_y}" z1="{zmin}" x2="{xmax}" y2="{ground_y+10}" z2="{zmax}" type="air"/>\n'
    
    # BORDER REMOVED - Just air as requested

    if random_placement:
        for _ in range(num_trees):
            x = random.randint(xmin + tree_radius, xmax - tree_radius)
            z = random.randint(zmin + tree_radius, zmax - tree_radius)
            # Randomly choose between birch and oak for a mixed forest
            tree_type = random.choice(["oak", "birch"])
            gen += draw_tree(x, ground_y, z, tree_radius, tree_type)
    else:
        trees_placed = 0
        for x in range(xmax-1, xmin, -1 * (2 * tree_radius + 1 + 1)):
            for z in range(zmax-1, zmin, -1 * (2 * tree_radius + 1 + 1)):
                if trees_placed < num_trees:
                    tree_type = random.choice(["oak", "birch"])
                    gen += draw_tree(x, ground_y, z, tree_radius, tree_type)
                    trees_placed += 1
    return gen

# Generate a FIXED dense forest of 40 mixed Oak/Birch trees in a 40x40 area.
tree_xml = draw_trees(25,-20, 20, -20, 20, random_placement=True)

missionXML = f'''<?xml version="1.0" encoding="UTF-8" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com">
    <About>
        <Summary>Fixed Mixed Forest VoE Test (No Borders)</Summary>
    </About>
    <ServerSection>
        <ServerHandlers>
            <FlatWorldGenerator generatorString="3;7,2*3,2;1;" forceReset="true"/>
            <DrawingDecorator>
                {tree_xml}
            </DrawingDecorator>
            <ServerQuitWhenAnyAgentFinishes/>
        </ServerHandlers>
    </ServerSection>
    <AgentSection mode="Survival">
        <Name>TreeBot</Name>
        <AgentStart>
            <Placement x="0.5" y="4.0" z="0.5" yaw="0"/>
        </AgentStart>
        <AgentHandlers>
            <DiscreteMovementCommands/>
            <VideoProducer want_depth="false">
                <Width>64</Width>
                <Height>64</Height>
            </VideoProducer>
        </AgentHandlers>
    </AgentSection>
</Mission>'''

# Create socket connection
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('localhost', 25565))
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

# Create default Malmo objects:
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)
print()
print("Mission running ")

setting = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
done = False

# Loop until mission ends:
while world_state.is_mission_running and not done:
    world_state = agent_host.getWorldState()
    while world_state.number_of_video_frames_since_last_state == 0 and world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        
    # If mission ended while waiting, break cleanly
    if not world_state.is_mission_running:
        done = True
    else:
        frame = prepare_video_data(world_state.video_frames[-1])
        action = get_LeWM_action(frame, connection, setting)
        action_list = process_LeWM_action(action, setting, True)
        if action_list:
            print(action_list)
            for act in action_list:
                agent_host.sendCommand(act)
        time.sleep(0.1)

exit_procedure()
print()
print("Mission ended")