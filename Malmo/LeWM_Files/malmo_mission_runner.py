from __future__ import print_function
from builtins import range
from lewm_integration_layer import get_LeWM_action, process_LeWM_action, prepare_video_data
import MalmoPython
import os
import sys
import time
import signal
import socket


if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

#   <Inventory>
#     <InventoryItem slot="0" type="diamond_axe"/>
#   </Inventory>

#       <!-- <FlatWorldGenerator generatorString="3;7,2*3,2;1;"/> -->
#   <DrawingDecorator>
#     <!-- {tree_xml} -->
#   </DrawingDecorator>

        # if setting != [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]: # For video generation, only save frames when the player is active
        #     for frame in world_state.video_frames[-1]:
        #         frame_data.append(prepare_video_data(frame))

        # with open("./artifacts/fixtures/video_frames.pkl", 'wb') as file:
#     print(type(frame_data))
#     pickle.dump(frame_data, file)


def draw_tree(x, y, z, r):
    gen = ""

    # Base leaf config
    leaf_br = r # Leaf base radius
    leaf_base_y = y + 2
    leaf_base_height = 2
    # Upper leaf config
    leaf_ur = 1 # Leaf upper radius
    leaf_upper_y = leaf_base_y + leaf_base_height
    leaf_upper_height = 2

    # Generate leaves
    gen += (f'<DrawCuboid x1="{x-leaf_br}" y1="{leaf_base_y}" z1="{z-leaf_br}" '
                        f'x2="{x+leaf_br}" y2="{leaf_base_y + leaf_base_height-1}" z2="{z+leaf_br}" type="leaves"/>\n') # Base
    gen += (f'<DrawCuboid x1="{x-leaf_ur}" y1="{leaf_upper_y}" z1="{z-leaf_ur}" '
                        f'x2="{x+leaf_ur}" y2="{leaf_upper_y + leaf_upper_height-1}" z2="{z+leaf_ur}" type="leaves"/>\n') # Upper leaves
    
    # Trim base leaves
    for i in range(leaf_base_y, leaf_base_y+leaf_base_height):
        gen += f'<DrawBlock x="{x+leaf_br}" y="{i}" z="{z+leaf_br}" type="air"/>\n'
        gen += f'<DrawBlock x="{x+leaf_br}" y="{i}" z="{z-leaf_br}" type="air"/>\n'
        gen += f'<DrawBlock x="{x-leaf_br}" y="{i}" z="{z+leaf_br}" type="air"/>\n'
        gen += f'<DrawBlock x="{x-leaf_br}" y="{i}" z="{z-leaf_br}" type="air"/>\n'

    # Trim upper leaves
    for i in range(leaf_upper_y+1, leaf_upper_y+leaf_upper_height):
        gen += f'<DrawBlock x="{x+leaf_ur}" y="{i}" z="{z+leaf_ur}" type="air"/>\n'
        gen += f'<DrawBlock x="{x+leaf_ur}" y="{i}" z="{z-leaf_ur}" type="air"/>\n'
        gen += f'<DrawBlock x="{x-leaf_ur}" y="{i}" z="{z+leaf_ur}" type="air"/>\n'
        gen += f'<DrawBlock x="{x-leaf_ur}" y="{i}" z="{z-leaf_ur}" type="air"/>\n'

    # Generate trunk
    for dy in range(y, leaf_upper_y + leaf_upper_height // 2):
        gen += f'<DrawBlock x="{x}" y="{dy}" z="{z}" type="log"/>\n'

    return gen


def draw_trees(num_trees, xmin, xmax, zmin, zmax, ground_y=4, random=True, border = False):
    gen = ""
    tree_radius=2
    tree_height=4

    # Clear original trees (SUPERFLAT ONLY)
    gen += (f'<DrawCuboid x1="{xmin-10}" y1="{ground_y}" z1="{zmin-10}" '
                        f'x2="{xmax+10}" y2="{ground_y + 30}" z2="{zmax+10}" type="air"/>\n')
    gen += (f'<DrawCuboid x1="{xmin-10}" y1="{ground_y-1}" z1="{zmin-10}" '
                        f'x2="{xmax+10}" y2="{ground_y-1}" z2="{zmax+10}" type="grass"/>\n')
    
    # If configured, add border to restrict environment
    if border:
        gen += (f'<DrawCuboid x1="{xmin-tree_radius-1}" y1="{ground_y}" z1="{zmin-tree_radius-1}" '
                        f'x2="{xmax+tree_radius+1}" y2="{ground_y+tree_height}" z2="{zmax+tree_radius+1}" type="barrier"/>\n')
        gen += (f'<DrawCuboid x1="{xmin-tree_radius}" y1="{ground_y}" z1="{zmin-tree_radius}" '
                        f'x2="{xmax+tree_radius}" y2="{ground_y+tree_height}" z2="{zmax+tree_radius}" type="air"/>\n')

    if random:
        for _ in range(num_trees):
            x = random.randint(xmin, xmax)
            z = random.randint(zmin, zmax)
            gen += draw_tree(x, ground_y, z, tree_radius)
    else:
        # x = xmin + tree_radius + 1
        # z = zmin + tree_radius + 1
        trees_placed = 0

        for x in range(xmax-1, xmin, -1 * (2 * tree_radius + 1 + 1)):
            for z in range(zmax-1, zmin, -1 * (2 * tree_radius + 1 + 1)):
                if trees_placed < num_trees:
                    gen += draw_tree(x, ground_y, z, tree_radius)
                    trees_placed += 1

    return gen


tree_xml = draw_trees(1, -5, 5, -5, 5, random=False, border=True) #draw_trees(25, -50, 50, -50, 50) # <- Random tree generation

missionXML = f'''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
        {tree_xml}
      </DrawingDecorator>
      <!-- <ServerQuitFromTimeUp timeLimitMs="60000"/> -->
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
            <Width>''' + str(64) + '''</Width>
            <Height>''' + str(64) + '''</Height>
        </VideoProducer>
      <ObservationFromFullStats/>
      <DiscreteMovementCommands/>
      <!-- <ContinuousMovementCommands turnSpeedDegs="180"/> -->
    </AgentHandlers>
  </AgentSection>

</Mission>
'''

# Create socket connection
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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

        for action in action_list:
            agent_host.sendCommand(action)

    time.sleep(0.1)

exit_procedure()
print()
print("Mission ended")
# Mission has ended.
