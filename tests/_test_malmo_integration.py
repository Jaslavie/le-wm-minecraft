from __future__ import print_function
from builtins import range
import MalmoPython
import os
import sys
import time

def get_LeWM_action():
    # Stub
    #[ forward, left, back, right, jump, sneak, sprint, attack , camera, camera]
    return [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]

def process_LeWM_action(action_list, current):
    # Convert array into instructions

    instructions = []

    # commands = ["move 1", "strafe -1", "move -1", "strafe 1", "jump 1", "crouch 1", "?????", "attack 1"]
    #            [ forward, left,         back,      right,      jump,     sneak,      sprint,  attack , camera, camera]

    for i, enabled in enumerate(action_list):
        if enabled != current[i]:
            current[i] = enabled

            # Forward/Backward Movement
            if i == 0 or i == 2:
                if action_list[0] == action_list[2]: # Opposite inputs cancel out
                    enabled = 0

                instructions.append(f"move {enabled}")

            # Left/Right Movement
            if i == 1 or i == 3:
                if action_list[1] == action_list[3]: # Opposite inputs cancel out
                    enabled = 0
                instructions.append(f"strafe -{enabled}")

            if i == 4:
                instructions.append(f"jump {enabled}")

            if i == 5:
                instructions.append(f"crouch {enabled}")

            if i == 6:
                pass # Sprint

            if i == 7:
                instructions.append(f"attack {enabled}")

            if i == 8:
                pass # TODO: Camera

            if i == 9:
                pass # TODO: Camera

    # instructions = ["move 1", "attack 1"] # ONLY HERE WHILE FUNCTION IS STUB

    return instructions

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

# More interesting generator string: "3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"
# Default generator string: "3;7,220*1,5*3,2;3;"
# <DrawingDecorator>
#   <DrawSphere x="-27" y="70" z="0" radius="30" type="air"/>
# </DrawingDecorator>
#

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Hello world!</Summary>
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
                  <!-- <DefaultWorldGenerator/> -->
                  <FlatWorldGenerator generatorString="3;7,2*3,2;4;"/>
                  <!-- <ServerQuitFromTimeUp timeLimitMs="30000"/> -->
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart>
                <!-- <Placement x="0" y="72" z="359"/> -->
                    <Inventory>
                        <InventoryItem slot="0" type="diamond_axe"/>
                    </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

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

# Loop until mission ends:
while world_state.is_mission_running:
    # for frame in world_state.video_frames:
    #     img = frame.pixels  # raw bytes
    #     width = frame.width
    #     height = frame.height
    # print(".", end="")

    action_list = get_LeWM_action() #input("Enter next action command: ")

    action_list = process_LeWM_action(action_list, setting)

    for action in action_list:
        agent_host.sendCommand(action)

    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission ended")
# Mission has ended.
