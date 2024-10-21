import os, json
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict, Counter

NO_ARG_ACTION_LIST = ["MoveAhead", "RotateRight",
                      "RotateLeft", "LookUp", "Done"]  # "LookDown"
OBJ_ARG_ACTION_LIST = ["PickupObject", "PutObject", "OpenObject", "CloseObject",
                       "SliceObject", "ToggleObjectOff", "ToggleObjectOn"]
direction_vectors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
degree_direction_mapping = {degree: direction_vectors
                            for degree, direction_vectors in zip(range(0, 360, 90), direction_vectors)}
direction_degree_mapping = {value: key for key, value in degree_direction_mapping.items()}
direction_text_mapping = {0: "north", 90: "east", 180: "south", 270: "west"}
text_direction_mapping = {"north": 0, "east": 90, "south": 180, "west": 270}

move_template = [
    "1) In the direction of my rotation, {:.0f} degrees ({}), the difference to the recommended position is {}={:.2f}m, meaning {:.2f}m {}. "
    "I need to move further {}.",
    "2) There is no obstacle in front of me in recent images.",
    "3) MoveAhead"]
end_rotate_template = [
    "1) My position is the same as the recommended one: {}. However, my rotation is {:.0f} degrees, facing {}. "
    "The recommended rotation is {:.0f} degrees, facing {}.",
    "2) Obstacles don't matter.",
    "3) {}"]
half_end_rotate_template = [
    "1) In the direction of my rotation, {:.0f} degrees ({}), the difference to the recommended position is {}={:.2f}m. "
    "Thus, I need to move in another direction, where the difference is {}={:.2f}m, meaning {:.2f}m ({}), and the rotation is {:.0f} degrees.",
    "2) Obstacles don't matter.",
    "3) {}"]
block_rotate_template = [
    "1) In the direction of my rotation {:.0f} degrees ({}), the difference compared to the recommended one is {}={:.2f}m.",
    "2) There are obstacles in front of me as shown in recent images. I need to rotate to another direction. "
    "In the other direction, the difference is {}={:.2f}m, meaning {:.2f}m ({}), and the rotation is {:.0f} degrees.",
    "3) {}"]
done_template = [
    "1) My position and rotation equal to the recommended one.",
    "2) I can see the target {} in the image of the current state.",
    "3) Done"]


def extract_object_type(object_id):
    # Split the string by '|'
    object_part = object_id.split('|')[0]
    # Split the object part by '-' and take the first part as object type
    object_type = object_part.split('-')[0]
    return object_type


def convert_pair_to_str(pair):
    return f"({pair[0]:.2f}, {pair[1]:.2f})"


def get_cur_diff(r, p, target_agent_pos):
    if r in {90, 270}:
        diff_len = target_agent_pos[0] - p[0]
        diff_direction = "east" if diff_len > 0 else "west"
        diff_eq = f"{target_agent_pos[0]:.2f}-{p[0]:.2f}"
    else:
        diff_len = target_agent_pos[1] - p[1]
        diff_direction = "north" if diff_len > 0 else "south"
        diff_eq = f"{target_agent_pos[1]:.2f}-{p[1]:.2f}"
    diff_len_abs = abs(diff_len)
    if diff_direction != direction_text_mapping[r]:
        print("Wrong cur direction")
    return diff_direction, diff_len, diff_len_abs, diff_eq


def get_other_diff(r, p, target_agent_pos, cur_action):
    # load other dir
    if r in {90, 270}:
        other_diff_len = target_agent_pos[1] - p[1]
        other_diff_direction = "north" if other_diff_len > 0 else "south"
        other_diff_eq = f"{target_agent_pos[1]:.2f}-{p[1]:.2f}"
    else:
        other_diff_len = target_agent_pos[0] - p[0]
        other_diff_direction = "east" if other_diff_len > 0 else "west"
        other_diff_eq = f"{target_agent_pos[0]:.2f}-{p[0]:.2f}"
    other_r = (r + 90 + 360) % 360 if cur_action == "RotateRight" else (r - 90 + 360) % 360
    other_diff_len_abs = abs(other_diff_len)
    if other_diff_direction != direction_text_mapping[other_r]:
        print("Wrong other direction")
    return other_r, other_diff_direction, other_diff_len, other_diff_len_abs, other_diff_eq


def add_cot_with_diff(cur_action, reachable_position_set, p, r, target_agent_pos,
            target_agent_r, idx, action_list, object_type):
    if cur_action == "MoveAhead":
        diff_direction, diff_len, diff_len_abs, diff_eq = get_cur_diff(r, p, target_agent_pos)
        step1 = move_template[0].format(r, direction_text_mapping[r], diff_eq, diff_len, diff_len_abs, diff_direction, diff_direction)
        step2 = move_template[1]
        step3 = move_template[2]
    elif cur_action == "RotateLeft" or cur_action == "RotateRight":
        # load reachable position
        move_d = degree_direction_mapping[r]
        next_p = (p[0] + move_d[0] * 0.25, p[1] + move_d[1] * 0.25)

        if idx == len(action_list) - 2:
            step1 = end_rotate_template[0].format(convert_pair_to_str(target_agent_pos),
                                                  r, direction_text_mapping[r],
                                                  target_agent_r, direction_text_mapping[target_agent_r])
            step2 = end_rotate_template[1]
            step3 = end_rotate_template[2].format(cur_action)
        elif r in {90, 270} and p[0] == target_agent_pos[0] or \
            r in {0, 180} and p[1] == target_agent_pos[1]:
            if r in {90, 270}:
                diff_len = target_agent_pos[0] - p[0]
                diff_eq = f"{target_agent_pos[0]:.2f}-{p[0]:.2f}"
            else:
                diff_len = target_agent_pos[1] - p[1]
                diff_eq = f"{target_agent_pos[1]:.2f}-{p[1]:.2f}"
            assert diff_len == 0
            other_r, other_diff_direction, \
                other_diff_len, other_diff_len_abs, other_diff_eq = get_other_diff(r, p, target_agent_pos, cur_action)
            step1 = half_end_rotate_template[0].format(r, direction_text_mapping[r], diff_eq, diff_len,
                                                       other_diff_eq, other_diff_len, other_diff_len_abs,
                                                       other_diff_direction, other_r)
            step2 = half_end_rotate_template[1]
            step3 = half_end_rotate_template[2].format(cur_action)
        elif next_p not in reachable_position_set:
            diff_direction, diff_len, diff_len_abs, diff_eq = get_cur_diff(r, p, target_agent_pos)
            other_r, other_diff_direction, \
                other_diff_len, other_diff_len_abs, other_diff_eq = get_other_diff(r, p, target_agent_pos, cur_action)
            step1 = block_rotate_template[0].format(r, direction_text_mapping[r], diff_eq, diff_len)
            step2 = block_rotate_template[1].format(other_diff_eq, other_diff_len, other_diff_len_abs, other_diff_direction, other_r)
            step3 = block_rotate_template[2].format(cur_action)
        else:
            raise ValueError("Wrong rotation")
    elif cur_action == "Done":
        step1 = done_template[0]
        step2 = done_template[1].format(object_type)
        step3 = done_template[2]
    else:
        raise ValueError("Wrong action")
    output_action = step1 + "\n" + step2 + "\n" + step3
    return output_action


def update_agent_status(position, rotation, action):
    position = deepcopy(position)
    if action == "RotateRight":
        rotation = (rotation + 90 + 360) % 360
    elif action == "RotateLeft":
        rotation = (rotation - 90 + 360) % 360
    elif action == "MoveAhead":
        cur_dv = degree_direction_mapping[rotation]
        position[0] += cur_dv[0] * 0.25
        position[1] += cur_dv[1] * 0.25
    return position, rotation


def get_potential_status(position, rotation):
    next_pos_list, next_r_list = [], []
    for action in ["MoveAhead", "RotateRight", "RotateLeft"]:
        next_pos, next_r = update_agent_status(position, rotation, action)
        next_pos_list.append(next_pos)
        next_r_list.append(next_r)
    return next_pos_list, next_r_list


def get_llava_format_data_with_diff(traj_dir, task_list, trial_num, sample_rate, image_num, action_his_num, use_cot):
    data_list = []
    prev_image_num = image_num - 1
    for task_name in tqdm(task_list, f"checking actions in train split"):
        # check the trial number
        scene_name, trial_name = task_name.split("-")
        if int(trial_name) >= trial_num:
            continue

        task_path = os.path.join(traj_dir, task_name)
        # start to sample examples
        traj_data = json.load(open(os.path.join(task_path, "traj.json")))
        action_list = traj_data["traj_info"]["action"]
        image_path_list = [f"{task_name}/images/i{idx}.png" for idx in range(len(action_list))]
        agent_pos_list, agent_r_list = traj_data["traj_info"]["corner"], traj_data["traj_info"]["rotation"]

        # object position, and get agent_pos_list and agent_r_list
        target_obj_pos = traj_data["objectInfo"]["position"]
        target_obj_pos = "({:.2f}, {:.2f})".format(target_obj_pos["x"], target_obj_pos["z"])
        target_agent_r = agent_r_list[-1]
        target_agent_pos = agent_pos_list[-1]

        if use_cot:
            with open(os.path.join(task_path, "reach_pos.json")) as fin:
                reachable_position_set = json.load(fin)
                reachable_position_set = set((pos[0], pos[1]) for pos in reachable_position_set)

        move_count = 0
        for idx, cur_action in enumerate(action_list):
            # cycling downsample
            if cur_action == "MoveAhead":
                move_count += 1
                if sample_rate != 1 and move_count % sample_rate != 1:
                    continue

            # organize the input/output of examples
            task_step_id = "{}_a{}".format(task_name, idx)
            # collect input information
            object_type = extract_object_type(traj_data["objectId"])
            next_pos_list, next_r_list = get_potential_status(agent_pos_list[idx], agent_r_list[idx])
            next_pos_list = ["({:.2f}, {:.2f})".format(pos[0], pos[1]) for pos in next_pos_list]
            next_r_list = ["{:.0f}".format(r) for r in next_r_list]
            # add history
            if action_his_num is None:
                history_str = []
            elif action_his_num > 0:
                his_action_list = action_list[max(idx - action_his_num, 0): max(idx - prev_image_num, 0)]
                his_pos_list = agent_pos_list[max(idx - action_his_num, 0): max(idx - prev_image_num, 0)]
                his_r_list = agent_r_list[max(idx - action_his_num, 0): max(idx - prev_image_num, 0)]
                history_str = [f"Position: ({p[0]:.2f}, {p[1]:.2f}), Rotation: {r:.0f}, Action: {a}"
                               for a, p, r in zip(his_action_list, his_pos_list, his_r_list)]
            else:
                raise KeyError(f"Wrong action_his_num value {action_his_num}")
            his_action_list = action_list[max(idx - prev_image_num, 0): idx]
            his_pos_list = agent_pos_list[max(idx - prev_image_num, 0): idx]
            his_r_list = agent_r_list[max(idx - prev_image_num, 0): idx]
            history_str += [f"Position: ({p[0]:.2f}, {p[1]:.2f}), Rotation: {r:.0f}, Current View: <image>, Action: {a}"
                            for a, p, r in zip(his_action_list, his_pos_list, his_r_list)]
            history_str = "\n".join(history_str)

            p, r = agent_pos_list[idx], agent_r_list[idx]
            cur_state_str = f"The current state is:\nPosition: ({p[0]:.2f}, {p[1]:.2f}), Rotation: {r:.0f}, Current View: <image>"

            # add cot prompt, add target position
            input_instruction = ("You are an agent placed in a 3D environment. "
                                 f"Your step length is 0.25 meters and rotation degree is 90. "
                                 f"You need to find a {object_type} at the position {target_obj_pos}. "
                                 f"To achieve this, we recommend you to move to the position ({target_agent_pos[0]:.2f}, "
                                 f"{target_agent_pos[1]:.2f}) with a rotation of {target_agent_r:.0f}.\n"
                                 f"Currently, you are at ({p[0]:.2f}, {p[1]:.2f}) with a rotation of {r:.0f}.\n\n"
                                 f"The possible actions are:\n"
                                 f"1. MoveAhead: Moves the agent forward by 0.25 meters in the direction it is currently facing. "
                                 f"For example, if the agent is at (x, y) facing 0 degrees (north), MoveAhead will result in (x, y + 0.25). "
                                 f"If the agent is facing 90 degrees (east), MoveAhead will result in (x + 0.25, y). "
                                 f"If the agent is facing 180 degrees (south), MoveAhead will result in (x, y - 0.25). "
                                 f"If the agent is facing 270 degrees (west), MoveAhead will result in (x - 0.25, y).\n"
                                 f"After MoveAhead in the current state, you will be at {next_pos_list[0]}.\n"
                                 f"2. RotateRight: Rotate right for 90 degrees (clockwise). "
                                 f"After RotateRight in current state, your rotation will be {next_r_list[1]} degrees.\n"
                                 f"3. RotateLeft: Rotate left for 90 degrees. (counterclockwise). "
                                 f"After RotateLeft in current state, your rotation will be {next_r_list[2]} degrees.\n"
                                 f"4. Done: Indicate that you are near to the target object and finish the task.\n\n"
                                 f"The history of recent states are:\n"
                                 f"{history_str}\n{cur_state_str}\n")
            if use_cot:
                input_instruction += (f"Please generate the next step given the above states with following steps: "
                                      f"1) Consider your rotation and position. 2) Check the images to see obstacles or "
                                      f"the target object."
                                      f"3) Decide the action.")
                output_action = add_cot_with_diff(cur_action, reachable_position_set, p, r, target_agent_pos,
                                        target_agent_r, idx, action_list, object_type)
            else:
                input_instruction += (f"Please generate the next step given the above states.")
                output_action = cur_action

            # we include the current image here
            example_dict = {"id": task_step_id, "image": image_path_list[max(idx - prev_image_num, 0): idx + 1],
                            "conversations": [{"from": "human", "value": input_instruction},
                                              {"from": "gpt", "value": output_action}]}
            data_list.append(example_dict)
    return data_list
