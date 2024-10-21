import os
import argparse
import numpy as np
import json
from transformers import set_seed
from typing import Dict, Any
from shapely import Polygon, Point

from tqdm import tqdm
from time import time
from utils import plot_top_down_frame, plot_house

from utils import init_controller, get_init_rotation
from utils import (load_dataset,
                   sample_init_position,
                   sample_task_params,
                   bfs_shortest_path_with_direction,
                   get_the_action_path,
                   collect_images,
                   init_dir_list,
                   collect_selected_object_type)
from utils import check_reverse_move
from utils import remove_traj_with_reverse_move
from utils import check_visible, check_rotation, check_easy_traj, load_scene_by_list


def get_rooms_polymap(house: Dict[str, Any]):
    room_poly_map = {}

    # NOTE: Map the rooms
    for i, room in enumerate(house["rooms"]):
        room_poly_map[room["id"]] = Polygon(
            [(p["x"], p["z"]) for p in room["floorPolygon"]]
        )

    return room_poly_map


def main(args):
    # init the controller
    set_seed(args.seed)
    start = time()
    controller = init_controller(args)
    end = time()
    print("init controller", end - start)

    with open(args.split_path) as fin:
        split_json = json.load(fin)
        valid_split = [scene_id for scene_id, _ in split_json["valid"]]
        test_split = [scene_id for scene_id, _ in split_json["test"]]
        eval_split = valid_split + test_split

    # init dataset and output dir
    train_dataset = load_scene_by_list(args.room_dir_path, eval_split)
    dataset_dir = args.save_path
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # start sampling
    pbar = tqdm(args.sample_amount, f"sampling {args.sample_amount} traces")
    count = args.start_idx
    while count < args.start_idx + args.sample_amount:
        success_flag = False
        room_id = eval_split[count]
        # refresh the controller every K scenes
        if (count + 1) % args.refresh_freq == 0:
            controller.stop()
            controller = init_controller(args)
        for task_id in range(args.trial_number):
            for retry_cnt in range(args.max_retry):
                # init the scene and agent in ai2thor
                set_seed(args.seed + count + task_id + retry_cnt)

                # init the folder and save all info
                room_save_path = os.path.join(dataset_dir, f"{room_id}-{task_id}")
                if os.path.exists(room_save_path) and os.path.exists(os.path.join(room_save_path, "reach_pos.json")):
                    success_flag = True
                    break
                else:
                    if not os.path.exists(room_save_path):
                        os.makedirs(room_save_path)

                print(f"{room_id} {task_id}, retry {retry_cnt}")
                room_json = train_dataset[room_id]
                controller.reset(scene=room_json)
                room_poly_dict = get_rooms_polymap(room_json)
                room_poly_keys = list(room_poly_dict.keys())

                # sample task position
                collected_object_set = collect_selected_object_type(dataset_dir, room_id, task_id)
                try:
                    (reachable_positions,
                     cur_object, target_position) = sample_task_params(controller, room_json,
                                                                       collected_object_set)
                except Exception as e:
                    print(f"task sample failed, task_id: {room_id}-{task_id}", e)
                    continue

                cur_object_room, cur_object_room_name = None, None
                for room_poly_key in room_poly_keys:
                    room_poly = room_poly_dict[room_poly_key]
                    cur_object_point = Point(target_position[0], target_position[1])
                    if room_poly.covers(cur_object_point):
                        cur_object_room_name, cur_object_room = room_poly_key, room_poly
                        break

                try:
                    # sample same-room init position here
                    init_p, move_count = sample_init_position(controller, target_position,
                                                              cur_object_room)
                except Exception as e:
                    print(f"init position sample failed, task_id: {room_id}-{task_id}", e)
                    continue

                # turn coordinates into 2D
                reachable_position_tuples = set((p[0], p[2]) for p in reachable_positions)
                target_position_tuple = (target_position[0], target_position[2])
                # compute the shortest path and action list
                try:
                    shortest_path = bfs_shortest_path_with_direction(reachable_position_tuples, init_p,
                                                                     target_position_tuple)
                except Exception as e:
                    print(f"no BFS path, task_id: {room_id}-{task_id}", e)
                    continue
                # truncate the movement reverse to the position of target_position_tuple
                shortest_path = remove_traj_with_reverse_move(init_p, target_position_tuple, shortest_path,
                                                              args.traj_max_len)
                try:
                    check_reverse_move(init_p, target_position_tuple, shortest_path)
                except Exception as e:
                    print(f"still reverse directions, task_id: {room_id}-{task_id}", e)
                    continue

                truncate_flag = shortest_path[0] != init_p

                # init agent with right rotation
                init_r = get_init_rotation(shortest_path)
                init_p = shortest_path[0]
                controller.step(
                    action="Teleport",
                    position=dict(x=init_p[0], y=0.95, z=init_p[1]),
                    rotation=dict(x=0, y=init_r, z=0),
                    horizon=30, standing=True)

                # finish task param sampling, start simulating
                init_agent_status = controller.last_event.metadata["agent"]
                try:
                    action_list, double_rotate_flag = get_the_action_path(shortest_path, init_agent_status, cur_object,
                                                                          retry_cnt)
                except Exception as e:
                    print(f"Two rotation failed, task_id: {room_id}-{task_id}", e)
                    continue
                if len(action_list) > 20:
                    print(f"Too long traj, task_id: {room_id}-{task_id}, length {len(action_list)}")
                    continue
                # conduct the action list and save images
                image_list, act_pos_list, act_r_list = [], [], []
                reachable_flag, double_visible_flag, end_idx, end_truncate_flag = True, False, None, False
                cur_image, depth_frame, instance_segment_frame = collect_images(controller)
                image_list.append((0, cur_image, depth_frame, instance_segment_frame))
                act_pos_list.append(init_p)
                act_r_list.append(init_r)
                for idx, a in enumerate(action_list[:-1]):  # use [:-1] since we don't need to include "Done"
                    controller.step(a)
                    if not controller.last_event.metadata["lastActionSuccess"]:
                        reachable_flag = False
                        break

                    cur_image, depth_frame, instance_segment_frame = collect_images(controller)
                    image_list.append((idx + 1, cur_image, depth_frame, instance_segment_frame))
                    new_p = controller.last_event.metadata["agent"]["position"]
                    new_p = [new_p["x"], new_p["z"]]
                    new_r = int(controller.last_event.metadata["agent"]["rotation"]["y"])
                    act_pos_list.append(new_p)
                    act_r_list.append(new_r)
                    if double_rotate_flag and idx >= len(action_list) - 6:  # five actions
                        double_visible_flag = check_visible(controller, cur_object)
                    if double_visible_flag:
                        print('truncate double rotate traj')
                        if idx != len(action_list) - 2:
                            end_truncate_flag = True
                        end_idx = idx
                        action_list = action_list[: end_idx + 1] + ["Done"]
                        target_position_tuple = (act_pos_list[-1][0], act_pos_list[-1][1])
                        break

                if not reachable_flag:
                    print(f"there are blocks on the route, task_id: {room_id}-{task_id}")
                    continue
                if action_list[-2].startswith("Rotate") and action_list[-3].startswith("Rotate"):
                    print(f"still two rotate, task_id: {room_id}-{task_id}")
                    continue
                try:
                    check_rotation(action_list, act_pos_list, act_r_list, reachable_position_tuples)
                except Exception as e:
                    print(f"invalid rotation, task_id: {room_id}-{task_id}", e)
                    continue
                try:
                    assert check_easy_traj(act_pos_list)
                except Exception as e:
                    print(f"still non easy traj, task_id: {room_id}-{task_id}", e)
                    continue

                cur_object_id = cur_object["objectId"]
                object_list = controller.last_event.metadata["objects"]
                check_object_status = None
                for obj in object_list:
                    if obj["objectId"] == cur_object_id:
                        check_object_status = obj
                        break
                try:
                    assert check_object_status is not None and check_object_status["visible"]
                except Exception as e:
                    print(f"object invisible, task_id: {room_id}-{task_id}", e)
                    continue
                # update action list

                end_agent_status = controller.last_event.metadata["agent"]

                task_info = {}
                task_info["room"] = f"{args.dataset}_{room_id}"
                task_info["task_id"] = task_id
                task_info["truncate_flag"] = truncate_flag
                task_info["end_truncate_flag"] = end_truncate_flag
                task_info["task_type"] = "nav"
                task_info["objectId"] = cur_object["objectId"]
                task_info["objectInfo"] = cur_object
                task_info["traj_info"] = {"action": action_list, "corner": act_pos_list,
                                          "rotation": act_r_list}
                task_info["shortest_path"] = shortest_path
                task_info["init_agent_status"] = init_agent_status
                task_info["end_agent_status"] = end_agent_status

                assert end_agent_status["position"]["x"] == act_pos_list[-1][0] and \
                       end_agent_status["position"]["z"] == act_pos_list[-1][1]
                assert len(action_list) == len(act_pos_list)

                # plot the top view
                plot_house(reachable_position_tuples, act_pos_list, room_save_path, target_position_tuple)
                plot_top_down_frame(controller, room_save_path)

                # save image
                i_dir, d_dir, s_dir = init_dir_list(room_save_path)
                depth_array = []
                for idx, cur_image, cur_depth, cur_segment in image_list:
                    cur_image.save(os.path.join(i_dir, f"i{idx}.png"))
                    cur_segment.save(os.path.join(s_dir, f"s{idx}.png"))
                    depth_array.append(cur_depth)
                np.save(os.path.join(d_dir, "depth.npy"), np.array(depth_array))

                # dump all the info
                with open(os.path.join(room_save_path, "traj.json"), "w") as fout:
                    json.dump(task_info, fout, indent=4)
                # dump all color info
                object_id_to_color = controller.last_event.object_id_to_color
                with open(os.path.join(room_save_path, "color.json"), "w") as fout:
                    json.dump(object_id_to_color, fout, indent=4)
                # dump all reachable position
                with open(os.path.join(room_save_path, "reach_pos.json"), "w") as fout:
                    json.dump(list(reachable_position_tuples), fout, indent=4)

                success_flag = True
                break

        if not success_flag:
            print(count, "out of retry times")
        count += 1
        pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default="../../new_trajectories/procthor_0_5",
                        help="where to save the generated data")
    parser.add_argument("--split_path", type=str, default="../../new_trajectories/procthor_room/split_file.json")
    parser.add_argument("--start_idx", type=int, default=35)
    parser.add_argument("--sample_amount", type=int, default=1000)
    parser.add_argument("--trial_number", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh_freq", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="procthor-10k")  # "holodeck")  # "procthor-10k")
    parser.add_argument("--max_retry", type=int, default=5)
    parser.add_argument("--headless", action="store_true", help="whether init the environment headlessly")
    parser.add_argument("--depth_image", action="store_true", help="whether render the depth image")
    parser.add_argument("--instance_segment", action="store_true", help="whether render the instance segmentation")
    parser.add_argument("--asset_dir", help="Directory to load assets from.",
                        default="../Holodeck/data/objaverse_holodeck/09_23_combine_scale"
                                "/processed_2023_09_23_combine_scale")
    parser.add_argument("--room_dir_path", type=str, default="../../new_trajectories/procthor_room")
    parser.add_argument("--traj_max_len", type=int, default=20)

    parse_args = parser.parse_args()
    print(parse_args)

    main(parse_args)
