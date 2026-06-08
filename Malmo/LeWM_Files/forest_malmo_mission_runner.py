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

def draw_tree(x, y, z, r):
    gen = ""
    leaf_br = r # Leaf base radius
    leaf_base_y = y + 2
    leaf_base_height = 2
    leaf_ur = 1 # Leaf upper radius
    leaf_upper_y = leaf_base_y + leaf_base_height
    leaf_upper_height = 2

    # Generate trunk
    for dy in range(y, leaf_upper_y + leaf_upper_height // 2):
        gen += f"<DrawBlock x='{x}' y='{dy}' z='{z}' type='log'/>\n"

    # Generate base leaves
    for dx in range(-leaf_br, leaf_br + 1):
        for dz in range(-leaf_br, leaf_br + 1):
            for dy in range(leaf_base_y, leaf_base_y + leaf_base_height):
                if abs(dx) == leaf_br and abs(dz) == leaf_br:
                    continue # Trim corners for rounder look
                gen += f"<DrawBlock x='{x+dx}' y='{dy}' z='{z+dz}' type='leaves'/>\n"

    # Generate upper leaves
    for dx in range(-leaf_ur, leaf_ur + 1):
        for dz in range(-leaf_ur, leaf_ur + 1):
            for dy in range(leaf_upper_y, leaf_upper_y + leaf_upper_height):
                gen += f"<DrawBlock x='{x+dx}' y='{dy}' z='{z+dz}' type='leaves'/>\n"
                
    return gen

def draw_trees(num_trees, xmin, xmax, zmin, zmax, ground_y=4, random_placement=True, border=False):
    gen = ""
    tree_radius = 2
    
    # If configured, add border to restrict environment
    if border:
        # FIX: Use BEDROCK so the agent cannot break the walls and escape
        gen += f"<DrawCuboid x1='{xmin}' y1='{ground_y}' z1='{zmin}' x2='{xmax}' y2='{ground_y+4}' z2='{zmin}' type='bedrock'/>\n"
        gen += f"<DrawCuboid x1='{xmin}' y1='{ground_y}' z1='{zmax}' x2='{xmax}' y2='{ground_y+4}' z2='{zmax}' type='bedrock'/>\n"
        gen += f"<DrawCuboid x1='{xmin}' y1='{ground_y}' z1='{zmin}' x2='{xmin}' y2='{ground_y+4}' z2='{zmax}' type='bedrock'/>\n"
        gen += f"<DrawCuboid x1='{xmax}' y1='{ground_y}' z1='{zmin}' x2='{xmax}' y2='{ground_y+4}' z2='{zmax}' type='bedrock'/>\n"

    if random_placement:
        for _ in range(num_trees):
            # Ensure trees don't spawn inside the bedrock walls
            x = random.randint(xmin + tree_radius + 1, xmax - tree_radius - 1)
            z = random.randint(zmin + tree_radius + 1, zmax - tree_radius - 1)
            gen += draw_tree(x, ground_y, z, tree_radius)
    else:
        trees_placed = 0
        for x in range(xmax - 1, xmin, -1 * (2 * tree_radius + 1 + 1)):
            for z in range(zmax - 1, zmin, -1 * (2 * tree_radius + 1 + 1)):
                if trees_placed < num_trees:
                    gen += draw_tree(x, ground_y, z, tree_radius)
                    trees_placed += 1
    return gen

# Generate a FIXED dense forest of 40 trees in a 40x40 area.
# random.seed(42) at the top ensures this layout is identical on every run.
tree_xml = draw_trees(40, -20, 20, -20, 20, random_placement=True, border=True)

# Reconstructed XML (The raw github link stripped the < > tags)
missionXML = f'''<?xml version="1.0" encoding="UTF-8" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com">
    <About>
        <Summary>Fixed Forest VoE Test</Summary>
    </About>
    <ServerSection>
        <ServerHandlers>
            <FlatWorldGenerator generatorString="3;7,2*3,2;1;"/>
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