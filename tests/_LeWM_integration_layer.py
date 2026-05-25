# from vit import tinyViT

def get_LeWM_action(action_count, n):
    # This function is currently a stub
    #[ forward, left, back, right, jump, sneak, sprint, attack , camera, camera]

    if action_count < n:
        action = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    else:
        action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    action_count += 1

    return action, action_count

def process_LeWM_action(action_list, current):
    # Convert array into instructions
    # commands = ["move 1", "strafe -1", "move -1", "strafe 1", "jump 1", "crouch 1", None,   "attack 1", "turn", "pitch"]
    #            [ forward, left,         back,      right,      jump,     sneak,     sprint,  attack , camera_x, camera_y]

    instructions = []

    move = action_list[0] - action_list [2]
    strafe = action_list[3] - action_list[1]

    if move != current[0] - current[2]:
        instructions.append(f"move {move}")
    
    if strafe != current[3] - current[1]:
        instructions.append(f"strafe {strafe}")

    if current[4] != action_list[4]:
        instructions.append(f"jump {action_list[4]}")

    if current[5] != action_list[5]:
        instructions.append(f"crouch {action_list[5]}")

    if current[7] != action_list[7]:
        instructions.append(f"attack {action_list[7]}")

    if action_list[8] != current[8]:
        instructions.append(f"turn {action_list[8]}")

    if action_list[9] != current[9]:
        instructions.append(f"pitch {action_list[9]}")

    for i, action in enumerate(action_list):
        current[i] = action

    return instructions

def prepare_video_data(frame_data):
    img = frame_data.pixels # Image pixel information: a flattened H X W X 3 array
    # width = frame_data.width
    # height = frame_data.height

    # TODO: Reshape to original image likeness if needed -- my understanding is that it isn't
    # frame = np.array(img)
    # frame.reshape((height, width, 3))

    # TODO: Any pre-processing for raw video pixels goes here
    # TODO: Run frames through encoder

    return img
