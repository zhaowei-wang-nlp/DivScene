import os
import sys
import json
import pickle
import argparse
from tqdm import tqdm
from PIL import Image
import requests
import base64
import time
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_data.utils import init_controller, load_scene_by_list
from sample_data.utils import extract_object_type, plot_house, plot_top_down_frame
from sample_data.gather_gpt4_prompt.completion_utils import retry_with_exponential_backoff
from reformat_data.utils import get_potential_status, add_cot

action_set = {"MoveAhead", "RotateRight",
              "RotateLeft", "Done"}
action_list = ["MoveAhead", "RotateRight",
              "RotateLeft", "Done"]
NO_ARG_ACTION_LIST = ["MoveAhead", "RotateRight",
                      "RotateLeft", "LookUp", "Done"]  # "LookDown"
a2i = {"moveahead": 0, "rotateright": 1,
       "rotateleft": 2, "done": 3}
''
def get_shortest_path_to_point(
    controller, initial_position, target_position
):
    kwargs = dict(
        action="GetShortestPathToPoint",
        position=initial_position,
        target=target_position
    )

    event = controller.step(kwargs)
    if event.metadata["lastActionSuccess"]:
        return event.metadata["actionReturn"]["corners"]
    else:
        raise ValueError(
            "Unable to find shortest path to point '{}'  due to error '{}'.".format(
                target_position, event.metadata["errorMessage"]
            )
        )


def map_action_to_id(action_list):
    id_list = [a2i.get(action.lower(), -1) for action in action_list]
    return id_list


def parse_action(output_str):
    output_str = output_str.strip()
    if output_str.startswith("Assistant:"):
        output_str = output_str[len("Assistant:"):].strip()
    try:
        output_str = output_str[output_str.index("3)") + 2:].strip()
    except Exception as e:
        pass
    output_str = output_str.split()
    for seg in output_str:
        if seg in action_set:
            return seg
    return "None"


def get_type_dict(my_dict):
    my_type_dict = {key: type(value) for key, value in my_dict.items()}
    return my_type_dict


def collect_images(controller):
    cur_image = Image.fromarray(controller.last_event.frame)

    return cur_image


@retry_with_exponential_backoff(initial_delay=1, exponential_base=1.05, jitter=False, max_retries=20)
def post_with_retry(url, cur_example):
    response = requests.post(url, json=cur_example)
    if response.status_code == 200:
        result = response.json()
    else:
        print(f"Wrong HTTP response code {response.status_code}")
        raise ValueError(f"Wrong HTTP response code {response.status_code}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="test")
    parser.add_argument("--dataset", type=str, default="holodeck")  # "procthor-10k")
    parser.add_argument("--headless", action="store_true", help="whether init the environment headlessly")
    parser.add_argument("--depth_image", action="store_true", help="whether render the depth image")
    parser.add_argument("--instance_segment", action="store_true", help="whether render the instance segmentation")
    parser.add_argument("--room_dir_path", type=str,
                        default="YOUR_PATH")
    parser.add_argument("--trial_dir_path", type=str, default="YOUR_PATH")
    parser.add_argument("--save_path", type=str,
                        default="YOUR_PATH")
    parser.add_argument("--freq_refresh", type=int, default=30)
    parser.add_argument("--image_num", type=int, help="the number of images",
                        default=4)
    parser.add_argument("--action_his_num", type=int, default=8, help="whether use full action history")
    parser.add_argument("--use_cot", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=27)
    parser.add_argument("--trial_list", type=str, default="0")
    parser.add_argument("--asset_dir", help="Directory to load assets from.",
                        default="YOUR_PATH/processed_2023_09_23_combine_scale")
    parser.add_argument("--max_action", type=int, default=60)
    parser.add_argument("--ip", type=str, default="11.255.125.249")
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--no_in_diff", type=int, default=0)
    parser.add_argument("--split_path", type=str, help="the path to the split file",
                        default="YOUR_PATH/split_file.json")
    args = parser.parse_args()
    args.save_path = f"{args.save_path}/{args.split}"
    args.trial_list = [int(trial_id) for trial_id in args.trial_list.split(",")]
    print(args)
    prev_image_num = args.image_num - 1

    # metric param
    succ_count, total_count = 0, 0

    # init controller and result saving-related vars
    controller = init_controller(args)
    # train/valid/test according to the split file
    # total 4600
    with open(args.split_path) as fin:
        split_json = json.load(fin)
    cur_split = split_json[args.split]
    cur_split = [idx for idx, _ in cur_split]

    dataset_split = load_scene_by_list(args.room_dir_path, cur_split)

    for scene_id in tqdm(cur_split[args.start_idx: args.end_idx],
                                                f"testing the {args.split} split"):
        scene_json = dataset_split[scene_id]
        if (scene_id + 1) % args.freq_refresh == 0:
            controller.stop()
            controller = init_controller(args)

        for trial_id in args.trial_list:
            with open(os.path.join(args.trial_dir_path, f"{scene_id}-{trial_id}", "traj.json")) as fin:
                traj_data = json.load(fin)
            task_save_path = os.path.join(args.save_path, f"{scene_id}-{trial_id}")
            total_count += 1

            if os.path.exists(task_save_path) and os.path.exists(os.path.join(task_save_path, "agent_traj.json")):
                with open(os.path.join(task_save_path, "agent_traj.json")) as fin:
                    agent_traj_data = json.load(fin)
                    succ_count += int(agent_traj_data["task_success"])
                    if not agent_traj_data["task_success"]:
                        print(f"{scene_id}-0")
                continue
            else:
                if not os.path.exists(task_save_path):
                    os.makedirs(task_save_path)

            controller.reset(scene=scene_json)
            init_p = traj_data["init_agent_status"]["position"]
            init_r = int(traj_data["init_agent_status"]["rotation"]["y"])
            controller.step(
                action="Teleport",
                position=dict(x=init_p["x"], y=0.95, z=init_p["z"]),
                rotation=dict(x=0, y=init_r, z=0),
                horizon=30, standing=True)

            try:
                SPL_shortest = get_shortest_path_to_point(controller, traj_data["init_agent_status"]["position"],
                                           traj_data["end_agent_status"]["position"])
                SPL_shortest = [(corner["x"], corner["z"]) for corner in SPL_shortest]
            except Exception as e:
                print("shortest path action failed.")
                SPL_shortest = traj_data["shortest_path"]
            init_agent_status = controller.last_event.metadata["agent"]
            target_object_id = traj_data["objectId"]
            # however, we only use object type to navigate
            object_type = extract_object_type(target_object_id)
            target_obj_pos = traj_data["objectInfo"]["position"]
            target_obj_pos = "({:.2f}, {:.2f})".format(target_obj_pos["x"], target_obj_pos["z"])

            target_agent_r = traj_data["traj_info"]["rotation"][-1]
            target_agent_pos = traj_data["traj_info"]["corner"][-1]

            print(f"scene split: {args.split} id: {scene_id}, trial_id: {trial_id}")
            print(f"init pos: {init_p}, target object type: {object_type}")
            action_count = 0
            all_image_list, all_res_list = [], []
            agent_action_list, agent_pos_list, agent_r_list = [], [], []
            previous_query, previous_action, loop_flag = None, None, False
            while action_count < args.max_action:
                example = {"id": f"{scene_id}-{trial_id}_a{len(agent_action_list)}"}

                # first update the image list
                cur_image = collect_images(controller)
                all_image_list.append(cur_image)
                # update pos and rotation
                cur_agent_state = controller.last_event.metadata["agent"]
                p = [cur_agent_state["position"]["x"], cur_agent_state["position"]["z"]]
                r = int(cur_agent_state["rotation"]["y"])
                agent_pos_list.append(p)
                agent_r_list.append(r)
                p_diff = (target_agent_pos[0] - p[0], target_agent_pos[1] - p[1])

                # get potential position list
                next_pos_list, next_r_list = get_potential_status(p, r)
                next_pos_list = ["({:.2f}, {:.2f})".format(pos[0], pos[1]) for pos in next_pos_list]
                next_r_list = ["{:.0f}".format(r) for r in next_r_list]

                # build history string part1
                a_idx = len(agent_action_list)
                action_his_num = args.action_his_num
                his_action_list = agent_action_list[max(a_idx - action_his_num, 0): max(a_idx - prev_image_num, 0)]
                his_pos_list = agent_pos_list[max(a_idx - action_his_num, 0): max(a_idx - prev_image_num, 0)]
                his_r_list = agent_r_list[max(a_idx - action_his_num, 0): max(a_idx - prev_image_num, 0)]
                history_str = [f"Position: ({p[0]:.2f}, {p[1]:.2f}), Rotation: {r:.0f}, Action: {a}"
                               for a, p, r in zip(his_action_list, his_pos_list, his_r_list)]
                # part2
                his_action_list = agent_action_list[max(a_idx - prev_image_num, 0): a_idx]
                his_pos_list = agent_pos_list[max(a_idx - prev_image_num, 0): a_idx]
                his_r_list = agent_r_list[max(a_idx - prev_image_num, 0): a_idx]
                history_str += [f"Position: ({p[0]:.2f}, {p[1]:.2f}), Rotation: {r:.0f}, Current View: <image>, Action: {a}"
                                for a, p, r in zip(his_action_list, his_pos_list, his_r_list)]
                history_str = "\n".join(history_str)
                cur_state_str = f"The current state is:\nPosition: ({p[0]:.2f}, {p[1]:.2f}), Rotation: {r:.0f}, Current View: <image>"

                if not args.no_in_diff:
                    input_instruction = ("You are an agent placed in a 3D environment. "
                                         f"Your step length is 0.25 meters and rotation degree is 90. "
                                         f"You need to find a {object_type} at the position {target_obj_pos}. "
                                         f"To achieve this, we recommend you to move to the position ({target_agent_pos[0]:.2f}, "
                                         f"{target_agent_pos[1]:.2f}) with a rotation of {target_agent_r:.0f}.\n"
                                         f"Currently, you are at ({p[0]:.2f}, {p[1]:.2f}) with a rotation of {r:.0f}. "
                                         f"The difference to the recommended position is ({p_diff[0]:.2f}, {p_diff[1]:.2f}).\n\n"
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
                else:
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
                if args.use_cot:
                    input_instruction += (f"Please generate the next step given the above states with following steps: "
                                          f"1) Consider your rotation and position. 2) Check the images to see obstacles or "
                                          f"the target object."
                                          f"3) Decide the action.")
                else:
                    # input_instruction += (f"Please generate the next step given the above states.")
                    input_instruction += (f"Please only generate the next step given the above states.")

                his_image_list = all_image_list[max(a_idx - prev_image_num, 0): a_idx + 1]
                his_image_list_bin = pickle.dumps(his_image_list)
                his_image_list_str = base64.b64encode(his_image_list_bin).decode('utf-8')
                example["image"] = his_image_list_str
                example["query"] = input_instruction

                print(example["id"])
                # print(get_type_dict(example))
                print("INPUT:", example["query"].split("\n")[-1])

                # add retry mechanism for post
                start_time = time.time()
                result = post_with_retry(f"http://{args.ip}:{args.port}/predict", example)
                end_time = time.time()
                print(end_time - start_time)


                # update action list, response list, and position list
                new_action = result["action"]
                if new_action == "None":
                    print(result["text"])
                    new_action = random.choice(action_list)
                controller.step(new_action)
                pos_after_action = controller.last_event.metadata["agent"]["position"]
                print("OUTPUT:", result, pos_after_action)
                result["id"] = example["id"]
                result["query"] = example["query"]
                all_res_list.append(result)
                agent_action_list.append(new_action)
                action_count += 1

                # check stop action "Done"
                if new_action == "Done":
                    break

                # check dead loop
                # check traj saving
                if previous_query is not None and previous_query == example["query"] and \
                    previous_action is not None and previous_action == result["action"]:
                    loop_flag = True
                    break
                previous_query = example["query"]
                previous_action = result["action"]
            end_agent_status = controller.last_event.metadata["agent"]
            object_list = controller.last_event.metadata["objects"]
            target_object_id = traj_data["objectId"]
            target_object_info = None
            task_success_flag = None
            for obj in object_list:
                if obj["objectId"] == target_object_id:
                    target_object_info = obj
                    task_success_flag = obj["visible"]
                    break
            assert task_success_flag is not None
            print(f"{args.dataset}_{scene_id}_{trial_id}", task_success_flag)
            succ_count += int(task_success_flag)



            # save the trace here
            agent_traj_data = {}
            agent_traj_data["room"] = f"{args.dataset}_{scene_id}"
            agent_traj_data["task_id"] = trial_id
            agent_traj_data["loop_flag"] = loop_flag
            agent_traj_data["task_type"] = "nav"
            agent_traj_data["task_success"] = task_success_flag
            agent_traj_data["objectId"] = target_object_id
            agent_traj_data["objectInfo"] = target_object_info
            agent_traj_data["traj_info"] = {"action": agent_action_list, "corner": agent_pos_list,
                                            "rotation": agent_r_list}
            agent_traj_data["res_list"] = all_res_list
            agent_traj_data["init_agent_status"] = init_agent_status
            agent_traj_data["end_agent_status"] = end_agent_status
            agent_traj_data["SPL_shortest"] = SPL_shortest

            # plot the top view
            event = controller.step(action="GetReachablePositions")
            reachable_position_tuples = set((p["x"], p["z"]) for p in event.metadata["actionReturn"])
            plot_house(reachable_position_tuples, agent_pos_list, task_save_path, agent_pos_list[-1])
            plot_top_down_frame(controller, task_save_path)

            image_dir_path = os.path.join(task_save_path, "images")
            if not os.path.exists(image_dir_path):
                os.makedirs(image_dir_path)

            for idx, cur_image in enumerate(all_image_list):
                cur_image.save(os.path.join(image_dir_path, f"i{idx}.png"))

            json.dump(agent_traj_data, open(os.path.join(task_save_path, "agent_traj.json"), "w"), indent=4)
            print(succ_count / total_count)
    print(succ_count / total_count)
