import sys, os, time, signal, socket, pickle, random
import MalmoPython
from lewm_integration_layer import get_LeWM_action, process_LeWM_action, prepare_video_data

def draw_tree(x, y, z, r):
    gen = ""
    leaf_base_y = y + 2
    leaf_base_height = 2
    leaf_upper_y = leaf_base_y + leaf_base_height
    leaf_upper_height = 2
    # Simplified XML generation for tree (matching repo logic)
    for dy in range(y, leaf_upper_y + leaf_upper_height // 2):
        gen += f"<DrawBlock x='{x}' y='{dy}' z='{z}' type='log'/>\n"
    for dx in range(-r, r+1):
        for dz in range(-r, r+1):
            for dy in range(leaf_base_y, leaf_base_y+leaf_base_height+2):
                if dx==0 and dz==0 and dy < leaf_base_y+2: continue
                gen += f"<DrawBlock x='{x+dx}' y='{dy}' z='{z+dz}' type='leaves'/>\n"
    return gen

def draw_trees(num_trees, xmin, xmax, zmin, zmax, ground_y=4, random_placed=True, border=False):
    gen = ""
    tree_radius = 2
    if not random_placed:
        trees_placed = 0
        for x in range(xmax-1, xmin, -1 * (2 * tree_radius + 1 + 1)):
            for z in range(zmax-1, zmin, -1 * (2 * tree_radius + 1 + 1)):
                if trees_placed < num_trees:
                    gen += draw_tree(x, ground_y, z, tree_radius)
                    trees_placed += 1
    return gen

def get_mission_xml(env_type):
    if env_type == "forest":
        # NO <Placement> tag. Malmo will use the natural safe spawn for this specific seed.
        return f'''<?xml version="1.0" encoding="UTF-8" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com">
            <About><Summary>Birch/Oak Forest VoE Test</Summary></About>
            <ServerSection>
                <ServerInitialConditions>
                    <Seed>-2744534680298546054</Seed>
                    <Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime></Time>
                </ServerInitialConditions>
                <ServerHandlers>
                    <DefaultWorldGenerator seed="-2744534680298546054"/>
                    <ServerQuitFromTimeUp timeLimitMs="60000"/>
                    <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
            </ServerSection>
            <AgentSection mode="Survival">
                <Name>VoE_Bot</Name>
                <AgentStart><Pitch>30</Pitch><Yaw>0</Yaw></AgentStart>
                <AgentHandlers>
                    <VideoProducer want_depth="false"><Width>64</Width><Height>64</Height></VideoProducer>
                    <DiscreteMovementCommands/>
                </AgentHandlers>
            </AgentSection>
        </Mission>'''
    else:
        tree_xml = draw_trees(1, -5, 5, -5, 5, random_placed=False, border=True)
        return f'''<?xml version="1.0" encoding="UTF-8" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com">
            <About><Summary>Single Tree Superflat</Summary></About>
            <ServerSection>
                <ServerHandlers>
                    <FlatWorldGenerator generatorString="3;7,2*3,2;1;clear" forceReset="true"/>
                    <DrawingDecorator>{tree_xml}</DrawingDecorator>
                    <ServerQuitFromTimeUp timeLimitMs="60000"/>
                    <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
            </ServerSection>
            <AgentSection mode="Survival">
                <Name>VoE_Bot</Name>
                <AgentStart><Placement x="0.5" y="4.0" z="3.5" pitch="30" yaw="0"/></AgentStart>
                <AgentHandlers>
                    <VideoProducer want_depth="false"><Width>64</Width><Height>64</Height></VideoProducer>
                    <DiscreteMovementCommands/>
                </AgentHandlers>
            </AgentSection>
        </Mission>'''

if __name__ == "__main__":
    env_type = sys.argv[1] if len(sys.argv) > 1 else "forest"
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 25565))
    server.listen(1)
    print(f"[Server] Listening for {env_type} connection...")
    connection, address = server.accept()

    agent_host = MalmoPython.AgentHost()
    my_mission = MalmoPython.MissionSpec(get_mission_xml(env_type), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    agent_host.startMission(my_mission, my_mission_record)

    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun: time.sleep(0.1); world_state = agent_host.getWorldState()

    setting = [0]*10
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        if world_state.number_of_video_frames_since_last_state > 0:
            frame = prepare_video_data(world_state.video_frames[-1])
            action = get_LeWM_action(frame, connection, setting)
            action_list = process_LeWM_action(action, setting, True)
            for act in action_list: agent_host.sendCommand(act)
        time.sleep(0.05)
    
    connection.close()
    server.close()