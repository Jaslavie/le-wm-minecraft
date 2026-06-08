#!/usr/bin/env python3
"""
Unified Malmo Runner for VoE Evaluation
Supports: Birch/Oak Forest (Custom Seed) and Single Tree Superflat.
Acts as a socket server to stream 64x64 frames to the VoE script.
"""

import MalmoPython
import time
import socket
import pickle
import argparse
import sys

def get_mission_xml(env_type, seed):
    if env_type == "forest":
        print(f"🌲 Generating Forest World with Seed: {seed}")
        return f'''<?xml version="1.0" encoding="UTF-8" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com">
          <About><Summary>VoE Forest</Summary></About>
          <ServerSection>
            <ServerInitialConditions>
                <Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime></Time>
                <Weather>clear</Weather>
            </ServerInitialConditions>
            <ServerHandlers>
              <DefaultWorldGenerator seed="{seed}"/>
              <ServerQuitFromTimeUp timeLimitMs="300000"/>
              <ServerQuitWhenAnyAgentFinishes/>
            </ServerHandlers>
          </ServerSection>
          <AgentSection mode="Survival">
            <Name>VoEBot</Name>
            <AgentStart><Placement x="0.5" y="70" z="0.5" yaw="0"/></AgentStart>
            <AgentHandlers>
              <VideoProducer want_depth="false">
                <Width>64</Width>
                <Height>64</Height>
              </VideoProducer>
              <DiscreteMovementCommands/>
            </AgentHandlers>
          </AgentSection>
        </Mission>'''
        
    elif env_type == "superflat":
        print("🟩 Generating Superflat World with Single Tree")
        tree_xml = '''
        <DrawCuboid x1="-2" y1="6" z1="-2" x2="2" y2="7" z2="2" type="leaves"/>
        <DrawCuboid x1="-1" y1="8" z1="-1" x2="1" y2="9" z2="1" type="leaves"/>
        <DrawBlock x="0" y="4" z="0" type="log"/>
        <DrawBlock x="0" y="5" z="0" type="log"/>
        <DrawBlock x="0" y="6" z="0" type="log"/>
        <DrawBlock x="0" y="7" z="0" type="log"/>
        '''
        return f'''<?xml version="1.0" encoding="UTF-8" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com">
          <About><Summary>VoE Superflat</Summary></About>
          <ServerSection>
            <ServerInitialConditions>
                <Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime></Time>
                <Weather>clear</Weather>
            </ServerInitialConditions>
            <ServerHandlers>
              <FlatWorldGenerator generatorString="3;7,2*3,2;1;"/>
              <DrawingDecorator>{tree_xml}</DrawingDecorator>
              <ServerQuitFromTimeUp timeLimitMs="300000"/>
              <ServerQuitWhenAnyAgentFinishes/>
            </ServerHandlers>
          </ServerSection>
          <AgentSection mode="Survival">
            <Name>VoEBot</Name>
            <AgentStart><Placement x="0.5" y="5" z="0.5" yaw="0"/></AgentStart>
            <AgentHandlers>
              <VideoProducer want_depth="false">
                <Width>64</Width>
                <Height>64</Height>
              </VideoProducer>
              <DiscreteMovementCommands/>
            </AgentHandlers>
          </AgentSection>
        </Mission>'''
    else:
        raise ValueError("env_type must be 'forest' or 'superflat'")

def parse_action_vector(vec):
    """Converts the 10-dim vector from the VoE script into Malmo discrete commands."""
    cmds = []
    # vec: [forward, left, back, right, jump, sneak, sprint, attack, camera_x, camera_y]
    if vec[0] == 1: cmds.append("move 1")
    elif vec[2] == 1: cmds.append("move -1")
    else: cmds.append("move 0")
    
    if vec[3] == 1: cmds.append("strafe 1")
    elif vec[1] == 1: cmds.append("strafe -1")
    else: cmds.append("strafe 0")
    
    if vec[4] == 1: cmds.append("jump 1")
    else: cmds.append("jump 0")
    
    if vec[8] != 0: cmds.append(f"turn {vec[8]}")
    if vec[9] != 0: cmds.append(f"pitch {vec[9]}")
    
    return cmds

def main():
    parser = argparse.ArgumentParser(description="Unified Malmo Runner for VoE")
    parser.add_argument("--env_type", type=str, required=True, choices=["forest", "superflat"])
    parser.add_argument("--seed", type=str, default="-2744534680298546054", help="Minecraft seed (used for forest)")
    parser.add_argument("--port", type=int, default=25565, help="Socket port to stream frames")
    args = parser.parse_args()

    # 1. Setup Malmo
    agent_host = MalmoPython.AgentHost()
    mission_xml = get_mission_xml(args.env_type, args.seed)
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    # Attempt to start mission
    for retry in range(3):
        try:
            agent_host.startMission(my_mission, my_mission_record)
            break
        except RuntimeError as e:
            if retry == 2:
                print("❌ Error starting mission:", e)
                return
            time.sleep(2)

    print("⏳ Waiting for Minecraft to load and mission to start...")
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        
    print("✅ Mission started!")

    # 2. Setup Socket Server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', args.port))
    server_socket.listen(1)
    
    print(f"🔌 Listening for VoE Client on localhost:{args.port}...")
    client_socket, addr = server_socket.accept()
    print(f"✅ Connected by {addr}. Streaming real frames...")

    # 3. Main Loop
    try:
        while world_state.is_mission_running:
            world_state = agent_host.getWorldState()
            
            # Send Frame to Client
            if world_state.number_of_video_frames_since_last_state > 0:
                frame = world_state.video_frames[-1]
                # frame.pixels is exactly 64*64*3 = 12288 bytes
                client_socket.sendall(frame.pixels)
                
                # Receive Action from Client
                data = client_socket.recv(1024)
                if data:
                    action_vec = pickle.loads(data)
                    commands = parse_action_vector(action_vec)
                    for cmd in commands:
                        agent_host.sendCommand(cmd)
            
            time.sleep(0.05) # Prevent CPU hogging
            
    except (ConnectionResetError, BrokenPipeError):
        print("⚠️ VoE Client disconnected.")
    except Exception as e:
        print(f"⚠️ Error: {e}")
    finally:
        client_socket.close()
        server_socket.close()
        print("🛑 Runner stopped.")

if __name__ == "__main__":
    main()
