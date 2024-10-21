from ai2thor.controller import Controller
import copy
import os
import math
import matplotlib.pyplot as plt
from typing import Dict, Any
import ai2thor.controller
from shapely import Polygon, Point
from shapely.ops import triangulate
import numpy as np
import json
import random
import heapq
from PIL import Image
from collections import deque
import prior
from copy import deepcopy
from numpy import linalg
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from ai2thor.platform import CloudRendering
from collections import defaultdict
from itertools import count

direction_vectors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
degree_direction_mapping = {degree: direction_vectors
                            for degree, direction_vectors in zip(range(0, 360, 90), direction_vectors)}
direction_degree_mapping = {value: key for key, value in degree_direction_mapping.items()}
degree_text_mapping = {0: "north", 90: "east", 180: "south", 270: "west"}
text_degree_mapping = {"north": 0, "east": 90, "south": 180, "west": 270}

# task constant params
visibilityDistance = 1.5
gridSize = 0.25
rotateStepDegrees = 90
yawDegree = 30
visiblePointNumber = min((2 * visibilityDistance / gridSize) ** 2, 10)


def get_init_rotation(shortest_path):
    for idx in range(len(shortest_path) - 1):
        pos1 = shortest_path[idx]
        pos2 = shortest_path[idx + 1]
        direction = (pos2[0] - pos1[0], pos2[1] - pos1[1])
        direction = (direction[0] / gridSize, direction[1] / gridSize)
        rotation = direction_degree_mapping[direction]
        return rotation


def create_rectangle(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    # Define the other two corners of the rectangle
    point3 = (x1, y2)
    point4 = (x2, y1)

    # Create the polygon (rectangle)
    rectangle = Polygon([point1, point3, point2, point4])

    return rectangle


def remove_traj_with_reverse_move(start_point, end_point, pos_list, max_len=15):
    # bounding_box = create_rectangle(start_point, end_point)
    # easy_traj = True
    # for pos in pos_list:
    #     pos = Point(pos)
    #     if not bounding_box.covers(pos):
    #         easy_traj = False
    #         break
    easy_traj = check_reverse_move(start_point, end_point, pos_list)

    if easy_traj:
        return pos_list
    r_count, his_direction = 0, None
    for idx in range(len(pos_list) - 1, 0, -1):
        cur_direction = (pos_list[idx][0] - pos_list[idx - 1][0],
                         pos_list[idx][1] - pos_list[idx - 1][1])
        if his_direction is not None and cur_direction != his_direction:
            r_count += 1
        his_direction = cur_direction
        if r_count == 2:
            break
    if len(pos_list) - idx > max_len:
        print(f"exceeding max length {max_len}")
    max_len_idx = max(len(pos_list) - max_len, 0)
    idx = max(idx, max_len_idx)
    pos_list = pos_list[idx:]
    return pos_list


def check_reverse_move(start_point, end_point, pos_list):
# get desirable direction
    direction_set = set()
    if end_point[0] > start_point[0]:
        direction_set.add("east")
    elif end_point[0] < start_point[0]:
        direction_set.add("west")
    if end_point[1] > start_point[1]:
        direction_set.add("north")
    elif end_point[1] < start_point[1]:
        direction_set.add("south")
    # check direction
    for idx in range(len(pos_list) - 1):
        cur_direction = (pos_list[idx + 1][0] - pos_list[idx][0],
                         pos_list[idx + 1][1] - pos_list[idx][1])
        cur_direction = (cur_direction[0] / gridSize, cur_direction[1] / gridSize)
        cur_degree = direction_degree_mapping[cur_direction]
        if degree_text_mapping[cur_degree] not in direction_set:
            return False
    return True


def remove_reverse_move_by_dir(start_point, end_point, pos_list, max_len=15):
    direction_set = set()
    if end_point[0] > start_point[0]:
        direction_set.add("east")
    elif end_point[0] < start_point[0]:
        direction_set.add("west")
    if end_point[1] > start_point[1]:
        direction_set.add("north")
    elif end_point[1] < start_point[1]:
        direction_set.add("south")
    # check direction
    easy_traj = True
    for idx in range(len(pos_list) - 1, 0, -1):
        cur_direction = (pos_list[idx][0] - pos_list[idx - 1][0],
                         pos_list[idx][1] - pos_list[idx - 1][1])
        cur_direction = (cur_direction[0] / gridSize, cur_direction[1] / gridSize)
        cur_degree = direction_degree_mapping[cur_direction]
        if degree_text_mapping[cur_degree] not in direction_set:
            easy_traj = False
            break
    if easy_traj:
        return pos_list

    if len(pos_list) - idx > max_len:
        print(f"exceeding max length {max_len}")
    max_len_idx = max(len(pos_list) - max_len, 0)
    idx = max(idx, max_len_idx)
    pos_list = pos_list[idx:]
    return pos_list


def check_easy_traj(pos_list):
    start_point, end_point = pos_list[0], pos_list[-1]
    bounding_box = create_rectangle(start_point, end_point)
    for pos in pos_list:
        pos = Point(pos)
        if not bounding_box.covers(pos):
            return False
    return True


def check_rotation(action_list, pos_list, r_list, reachable_position_set):
    target_pos = pos_list[-1]
    for idx, action in enumerate(action_list):
        if not action.startswith("Rotate"):
            continue
        cur_r = r_list[idx]
        cur_p = pos_list[idx]
        move_d = degree_direction_mapping[cur_r]
        next_p = (cur_p[0] + move_d[0] * 0.25, cur_p[1] + move_d[1] * 0.25)
        if idx == len(action_list) - 2:
            continue
        if next_p not in reachable_position_set:
            continue
        if (cur_r == 0 or cur_r == 180) and cur_p[1] == target_pos[1] or \
            (cur_r == 90 or cur_r == 270) and cur_p[0] == target_pos[0]:
            continue
        raise ValueError("step: " + str(idx) + " in " + str(len(action_list)))


def extract_object_type(object_id):
    # Split the string by '|'
    object_part = object_id.split('|')[0]
    # Split the object part by '-' and take the first part as object type
    object_type = object_part.split('-')[0]
    return object_type


def collect_selected_object_type(dataset_dir, room_id, task_id, trial_start=0):
    selected_object_set = set()
    for i in range(trial_start, task_id):
        prev_task_path = os.path.join(dataset_dir, f"{room_id}-{i}")
        try:
            with open(os.path.join(prev_task_path, "traj.json")) as fin:
                prev_traj_data = json.load(fin)
                prev_object_id = extract_object_type(prev_traj_data["objectId"])
                selected_object_set.add(prev_object_id)
        except Exception as e:
            print("!!!!!!!!!!!!!!! traj missing", e)
            print("missing path", os.path.join(prev_task_path, "traj.json"))
    return selected_object_set


def categorize_object_by_type(object_list):
    object_category = defaultdict(list)
    for obj in object_list:
        object_id = obj["objectId"]
        object_type = extract_object_type(object_id)
        object_category[object_type].append(obj)
    return object_category


def split_category(used_type_set, obj_category):
    new_obj_cate, used_obj_cate = {}, {}
    for obj_type, obj_list in obj_category.items():
        if obj_type in used_type_set:
            used_obj_cate[obj_type] = obj_list
        else:
            new_obj_cate[obj_type] = obj_list
    return new_obj_cate, used_obj_cate


def init_object_sampler(new_obj_cate, used_obj_cate):
    for obj_cate in [new_obj_cate, used_obj_cate]:
        key_list = list(obj_cate.keys())
        random.shuffle(key_list)
        for obj_type in key_list:
            obj_list = obj_cate[obj_type]
            random.shuffle(obj_list)
            for cur_obj in obj_list:
                yield cur_obj


def init_controller(args):
    need_objaverse = args.dataset == "holodeck"
    if not need_objaverse:
        controller = Controller(platform=CloudRendering if args.headless else None, renderDepthImage=args.depth_image,
                                renderInstanceSegmentation=args.instance_segment)
    else:
        controller = Controller(platform=CloudRendering if args.headless else None, renderDepthImage=args.depth_image,
                                renderInstanceSegmentation=args.instance_segment,
                                action_hook_runner=ProceduralAssetHookRunner(asset_directory=args.asset_dir,
                                                                             verbose=True))
    return controller


def manhattan_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return abs(x2 - x1) + abs(y2 - y1)


def sample_init_position(controller, target_position, cur_object_room, shortest_path=5, longest_path=12):
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]
    if len(reachable_positions) <= 30:
        print(f"{len(reachable_positions)} position. Space is very small!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    near_positions, new_reachable_position = [], []
    target_position = (target_position[0], target_position[2])
    for p in reachable_positions:
        p = (p["x"], p["z"])
        dis = manhattan_distance(p, target_position)
        move_count = dis / 0.25
        if move_count < shortest_path or move_count > longest_path:
            continue
        # point = Point(p[0], p[1])
        # if not cur_object_room.covers(point):
        #     continue
        near_positions.append([p, move_count])
        new_reachable_position.append(p)
    reachable_positions = new_reachable_position
    # p_set = set(reachable_positions)
    random.shuffle(near_positions)

    if near_positions:
        init_p, move_count = None, None
        for init_p, move_count in near_positions:
            controller.step(
                action="Teleport",
                position=dict(x=init_p[0], y=0.95, z=init_p[1]), horizon=30, standing=True)
            if 0.9 < controller.last_event.metadata["agent"]["position"]["y"] < 1.0 and \
                abs(init_p[0] - controller.last_event.metadata["agent"]["position"]["x"]) < 0.1 and \
                abs(init_p[1] - controller.last_event.metadata["agent"]["position"]["z"]) < 0.1:
                break
        return init_p, move_count
    else:
        raise ValueError("No enough position for init")
    # init_p = None
    # for cur_p, move_count in near_positions:
    #     nei_pos = []
    #     for d in direction_vectors:
    #         neighbor = (cur_p[0] + d[0] * gridSize, cur_p[1] + d[1] * gridSize)
    #         nei_pos.append(neighbor in p_set)
    #     if all(nei_pos):
    #         init_p = cur_p
    #         break


def init_dir_list(room_save_path):
    image_dir_path = os.path.join(room_save_path, "images")
    depth_dir_path = os.path.join(room_save_path, "depth")
    segment_dir_path = os.path.join(room_save_path, "segment")
    if not os.path.exists(image_dir_path):
        os.makedirs(image_dir_path)
    if not os.path.exists(depth_dir_path):
        os.makedirs(depth_dir_path)
    if not os.path.exists(segment_dir_path):
        os.makedirs(segment_dir_path)
    return image_dir_path, depth_dir_path, segment_dir_path


def collect_images(controller):
    cur_image = Image.fromarray(controller.last_event.frame)
    depth_frame = controller.last_event.depth_frame
    instance_segment = Image.fromarray(controller.last_event.instance_segmentation_frame)
    return cur_image, depth_frame, instance_segment


def load_dataset(dataset_name, args):
    if dataset_name == "procthor-10k":
        dataset = prior.load_dataset(dataset_name)
        dataset, valid_dataset, test_dataset = dataset["train"], dataset["val"], dataset["test"]
    elif dataset_name == "holodeck":
        dataset = load_scene_dir(args.room_dir_path)
    else:
        raise ValueError("Unsupported dataset name")
    return dataset


def load_scene_dir(dir_path, start_idx=0, end_idx=None):
    scene_name_list = os.listdir(dir_path)
    scene_dict = {}
    for scene_name in scene_name_list:
        scene_path = os.path.join(dir_path, scene_name)
        if not os.path.isdir(scene_path):
            continue
        scene_id = int(scene_name.split("_")[0])

        if end_idx is not None and (
            scene_id < start_idx or scene_id >= end_idx
        ):
            continue

        for file in os.listdir(scene_path):
            if file.endswith(".json"):
                scene_path = os.path.join(scene_path, file)
                break
        with open(scene_path) as fin:
            room_json = json.load(fin)
            scene_dict[scene_id] = room_json
    return scene_dict


def load_scene_by_list(dir_path, scene_id_list):
    scene_id_set = set(scene_id_list)
    scene_name_list = os.listdir(dir_path)
    scene_dict = {}
    for scene_name in scene_name_list:
        scene_path = os.path.join(dir_path, scene_name)
        if not os.path.isdir(scene_path):
            continue
        scene_id = int(scene_name.split("_")[0])

        if scene_id not in scene_id_set:
            continue

        for file in os.listdir(scene_path):
            if file.endswith(".json"):
                scene_path = os.path.join(scene_path, file)
                break
        with open(scene_path) as fin:
            room_json = json.load(fin)
            scene_dict[scene_id] = room_json
    return scene_dict


def sample_direction():
    direction_action = ["RotateRight", "RotateLeft"]
    selected_action = random.sample(direction_action, k=1)[0]
    return selected_action


def get_the_action_path(shortest_path, init_agent_status, target_object, retry_cnt):
    agent_rotation = init_agent_status["rotation"]["y"]

    action_list = []
    for i in range(1, len(shortest_path)):
        step_diff = (shortest_path[i][0] - shortest_path[i - 1][0],
                     shortest_path[i][1] - shortest_path[i - 1][1])
        tar_direction = tuple(axis / gridSize for axis in step_diff)
        assert tar_direction in direction_vectors
        tar_rotation = direction_degree_mapping[tar_direction]
        if tar_rotation == agent_rotation:
            action_list.append("MoveAhead")
        else:
            right_rotation = (tar_rotation - agent_rotation + 360) % 360
            if right_rotation < 180:
                agent_rotation = (agent_rotation + 90) % 360
                action_list.append("RotateRight")
            elif right_rotation > 180:
                agent_rotation = (agent_rotation - 90) % 360
                action_list.append("RotateLeft")
            else:
                agent_rotation = (agent_rotation + 180) % 360
                selected_direction = sample_direction()
                action_list.append(selected_direction)
                action_list.append(selected_direction)
                print("-----------------------------------two rotate")
            action_list.append("MoveAhead")
    # we need to change the rotation of the agent
    # here, we compute the rotation of the target object relative to the agent
    # at the last position of the shortest path.
    degree = get_relative_degree((target_object["position"]["x"], target_object["position"]["z"]),
                                 shortest_path[-1])
    object_rotation = convert_degree_to_direction(degree)
    double_rotate_flag = False
    if object_rotation != agent_rotation:
        right_rotation = (object_rotation - agent_rotation + 360) % 360
        if right_rotation < 180:
            action_list.append("RotateRight")
        elif right_rotation > 180:
            action_list.append("RotateLeft")
        else:
            print("Wrong layout, need double rotate.")
            selected_direction = sample_direction()
            action_list.append(selected_direction)
            action_list.append(selected_direction)
            double_rotate_flag = True

    # if target_object["position"]["y"] >= init_agent_status["position"]["y"]:
    #     action_list.append("LookUp")

    action_list.append("Done")
    return action_list, double_rotate_flag


def check_visible(controller, cur_object):
    cur_object_id = cur_object["objectId"]
    object_list = controller.last_event.metadata["objects"]
    check_object_status = False
    for obj in object_list:
        if obj["objectId"] == cur_object_id:
            check_object_status = obj
            break
    check_object_status = check_object_status["visible"]
    return check_object_status


def sample_task_params(controller, room_json, used_type_set):
    object_list = deepcopy(controller.last_event.metadata["objects"])
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]
    reachable_positions = convert_point_list_to_matrix(reachable_positions)
    random.shuffle(object_list)
    new_object_list = []
    for obj in object_list:
        if not check_object_qualify(obj):
            continue
        new_object_list.append(obj)
    object_list = new_object_list

    # classify object by type
    obj_category = categorize_object_by_type(object_list)
    # split used object type and new type
    new_obj_cate, used_obj_cate = split_category(used_type_set, obj_category)
    # init a object sampler
    object_sampler = init_object_sampler(new_obj_cate, used_obj_cate)

    selected_object, selected_position = None, None
    for anchor_object in object_sampler:
        object_position = convert_point_list_to_matrix([anchor_object["position"]])
        distance_list = linalg.norm(reachable_positions - object_position, axis=1)
        # partition top_k values, then sort them (this is faster than directly sorting them
        cur_visiblePointNumber = min(visiblePointNumber, len(distance_list))
        # the partition functions take index of elements. So, for the K-th element, we input
        # K - 1. Then, we slice the first K elements
        smallest_K_indices = np.argpartition(distance_list, cur_visiblePointNumber - 1)[:cur_visiblePointNumber]
        smallest_K_position = reachable_positions[smallest_K_indices]
        smallest_K_distance = distance_list[smallest_K_indices]
        # sort the smallest K positions
        sorted_smallest_K_indices = np.argsort(smallest_K_distance)
        smallest_K_position = smallest_K_position[sorted_smallest_K_indices]
        smallest_K_distance = smallest_K_distance[sorted_smallest_K_indices]
        pos_start_idx = random.randint(1, 5)
        selected_position = None
        for cur_p, cur_d in zip(smallest_K_position[pos_start_idx:], smallest_K_distance[pos_start_idx:]):
            if cur_d >= visibilityDistance:
                break
            rel_degree = get_relative_degree(object_position[0], cur_p)
            rel_rotation = convert_degree_to_direction(rel_degree)
            # rel_yaw_degree = get_yaw_degree(object_position[0], cur_p)

            # we first teleport the agent to see whether the position is valid
            # cases solved by this method: 1) the object is too low to see
            # 2) there is a wall between the agent's final position and the target object
            event = controller.step(
                action="Teleport",
                position=dict(x=cur_p[0], y=cur_p[1], z=cur_p[2]),
                rotation=dict(x=0, y=rel_rotation, z=0),
                horizon=30,
                standing=True
            )
            if not event.metadata["lastActionSuccess"]:
                continue
            visible_object_list = collect_visible_objects(event.metadata["objects"], used_type_set)
            if visible_object_list:
                selected_object = visible_object_list[0]
                selected_position = cur_p
                break
        if selected_position is not None:
            break
    if selected_object is None or selected_position is None:
        raise ValueError("Cannot find a proper target object")
    # we teleport the agent, when finding possible objects
    # Thus, we need to reset the room
    controller.reset(scene=room_json)
    # controller.step(
    #     action="Teleport",
    #     position=dict(x=init_p[0], y=0.95, z=init_p[1]), horizon=30, standing=True)
    return reachable_positions, selected_object, selected_position


# def is_safe(cur_point, reachable_positions, buffer=2):
#     """ Check if the neighbor has at least 'buffer' reachable positions around it. """
#     for d in direction_vectors:
#         neighbor = (cur_point[0] + d[0] * gridSize * buffer, cur_point[1] + d[1] * gridSize * buffer)
#         if neighbor not in reachable_positions:
#             return False
#     return True
# boundary x


def get_priority_directions(current_direction):
    """
    Return direction vectors with priority given to the current direction.
    """
    if current_direction is None:
        return direction_vectors
    # Create a priority list based on the current direction
    priority_directions = [current_direction]
    for d in direction_vectors:
        if d != current_direction:
            priority_directions.append(d)
    return priority_directions


def bfs_shortest_path_with_direction(reachable_positions, start, goal):
    # Priority queue: stores (cost, point, direction)
    queue = []
    counter = count()
    heapq.heappush(queue, (0, next(counter), start, None))
    visited = {start: 0}
    parents = {start: None}

    while queue:
        cost, _, current, direction = heapq.heappop(queue)

        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parents[current]
            return path[::-1]  # reverse path

        for d in get_priority_directions(direction):  # iterate by current direction
            neighbor = (current[0] + d[0] * gridSize, current[1] + d[1] * gridSize)
            if neighbor not in reachable_positions:
                continue
            move_cost = 1
            # this make sure that we find the path with least rotation
            if direction != d:
                move_cost += 1

            # this make sure that we find the path that rotate as late as possible
            total_cost = cost + move_cost
            if neighbor not in visited or total_cost < visited[neighbor]:
                visited[neighbor] = total_cost
                parents[neighbor] = current
                heapq.heappush(queue, (total_cost, next(counter), neighbor, d))

    raise ValueError("No path to the target")  # If no path found


def bfs_shortest_path(reachable_positions, cur_object_position, sampled_target_position):
    queue = deque([cur_object_position])
    parents = {cur_object_position: None}
    visited = set([cur_object_position])

    while queue:
        node = queue.popleft()

        if node == sampled_target_position:
            # Reconstruct the path from goal to start
            path = []
            while node is not None:
                path.append(node)
                node = parents[node]
            return path[::-1]  # Reverse the path to get the correct order

        for d in direction_vectors:
            neighbor = (node[0] + d[0] * gridSize, node[1] + d[1] * gridSize)
            if neighbor not in visited and neighbor in reachable_positions:
                visited.add(neighbor)
                parents[neighbor] = node
                queue.append(neighbor)

    return None  # No path found


def round_to_quarter(num, direction='nearest'):
    if direction == 'up':
        return math.ceil(num / 0.25) * 0.25
    elif direction == 'down':
        return math.floor(num / 0.25) * 0.25
    else:
        return round(num / 0.25) * 0.25


def get_rooms_polymap(house: Dict[str, Any]):
    room_poly_map = {}

    # NOTE: Map the rooms
    for i, room in enumerate(house["rooms"]):
        room_poly_map[room["id"]] = Polygon(
            [(p["x"], p["z"]) for p in room["floorPolygon"]]
        )

    return room_poly_map


def check_polygon_overlaps(room_poly_map):
    keys = list(room_poly_map.keys())
    overlaps = []

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            poly1 = room_poly_map[keys[i]]
            poly2 = room_poly_map[keys[j]]
            intersection = poly1.intersection(poly2)
            if intersection.area > 0:
                overlaps.append((keys[i], keys[j]))

    return overlaps


def get_candidate_points_in_room(
    room_id: str,
    room_poly_map: Dict[str, Polygon],
):
    polygon = room_poly_map[room_id]

    room_triangles = triangulate(polygon)

    candidate_points = [
        ((t.centroid.x, t.centroid.y), t.area) for t in room_triangles  # type:ignore
    ]

    # We sort the triangles by size so we try to go to the center of the largest triangle first
    candidate_points.sort(key=lambda x: x[1], reverse=True)
    candidate_points = [p[0] for p in candidate_points]

    # The centroid of the whole room polygon need not be in the room when the room is concave. If it is,
    # let's make it the first point we try to navigate to.
    if polygon.contains(polygon.centroid):
        candidate_points.insert(0, (polygon.centroid.x, polygon.centroid.y))

    return candidate_points


def my_get_candidate_points_in_room(
    room_id: str,
    room_poly_map: Dict[str, Polygon],
):
    polygon = room_poly_map[room_id]

    grid_size = 0.5

    # Get the bounding box of the polygon
    min_x, min_y, max_x, max_y = polygon.bounds
    min_x, min_y = round_to_quarter(min_x, "up"), round_to_quarter(min_y, "up")
    max_x, max_y = round_to_quarter(max_x, "down"), round_to_quarter(max_y, "down")

    # Generate grid points
    x_coords = np.arange(min_x + grid_size, max_x, grid_size)
    y_coords = np.arange(min_y + grid_size, max_y, grid_size)

    # Check each point to see if it's inside the polygon
    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)
            if polygon.contains(point):
                yield x, y


def try_find_collision_free_starting_position(
    house: Dict[str, Any],
    controller: ai2thor.controller.Controller,
    room_poly_map: Dict[str, Polygon],
):
    teleport_success, max_pos_num, selected_cand = False, -1e9, None
    for room_id in sorted(room_poly_map.keys()):
        candidate_points = my_get_candidate_points_in_room(room_id, room_poly_map)
        for cand in candidate_points:
            event = controller.step(
                action="TeleportFull",
                position={
                    "x": float(cand[0]),
                    "y": 0.95,
                    "z": float(cand[1]),
                },
                rotation=house["metadata"]["agent"]["rotation"],
                standing=True,
                horizon=30,
            )
            if event:
                event = controller.step("GetReachablePositions")
                reachable_positions = event.metadata["actionReturn"]
                if len(reachable_positions) >= max_pos_num:
                    selected_cand = cand
                    max_pos_num = len(reachable_positions)
                if max_pos_num <= 30:
                    continue
                else:
                    print(max_pos_num)
                    teleport_success = True
                    break

        if teleport_success:
            break

    return selected_cand


def check_room_size(
    room_poly_map: Dict[str, Polygon],
):
    total_size = 0
    for room_id in sorted(room_poly_map.keys()):
        cur_room = room_poly_map[room_id]
        total_size += cur_room.area

    return total_size


def plot_house(reachable_positions, shortest_path, room_save_path, target_position_tuple,
               name="grid_map.png"):
    xs = [rp[0] for rp in reachable_positions]
    zs = [rp[1] for rp in reachable_positions]

    xp = [rp[0] for rp in shortest_path]
    zp = [rp[1] for rp in shortest_path]
    fig, ax = plt.subplots()
    ax.scatter(xs, zs, s=100)  # , color='blue'
    ax.scatter(xp, zp, s=100)  # , color='red'
    ax.scatter([xp[0]], [zp[0]], color='red', s=100)
    ax.scatter([target_position_tuple[0]], [target_position_tuple[1]], color='purple', s=100)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_title("Reachable Positions in the Scene")
    ax.set_aspect("equal")
    # plt.show()
    plt.savefig(os.path.join(room_save_path, name))
    plt.close()


def get_relative_degree(position1, position2):
    if len(position1) == 3:
        position1 = (position1[0], position1[2])
    if len(position2) == 3:
        position2 = (position2[0], position2[2])
    x, y = position1[0] - position2[0], \
           position1[1] - position2[1]
    degree = (math.atan2(x, y) / math.pi * 180 + 360) % 360
    return degree


def get_yaw_degree(position1, position2):
    p1_height = position1[1]
    p2_height = position2[1]
    if p1_height >= p2_height:
        yaw_degree = 0
    else:
        yaw_degree = 30
    return yaw_degree


def check_object_visible(object_list, object_id):
    visible = False
    for obj in object_list:
        if obj["objectId"] == object_id:
            visible = obj["visible"]
    return visible


def collect_visible_objects(object_list, used_type_set):
    new_obj_list, used_obj_list = [], []
    for obj in object_list:
        if obj["visible"] and check_object_qualify(obj):
            obj_type = extract_object_type(obj["objectId"])
            if obj_type in used_type_set:
                used_obj_list.append(obj)
            else:
                new_obj_list.append(obj)
    random.shuffle(new_obj_list), random.shuffle(used_obj_list)
    visible_obj_list = new_obj_list + used_obj_list
    return visible_obj_list


def check_object_qualify(cur_obj):
    try:
        corner_point_list = np.array(cur_obj['axisAlignedBoundingBox']["cornerPoints"])
        point_height_list = corner_point_list[:, 1]
        max_height, min_height = np.max(point_height_list), np.min(point_height_list)
    except Exception as e:
        print(e)
        max_height, min_height = cur_obj["position"]["y"], cur_obj["position"]["y"]

    if cur_obj["objectId"].startswith("wall|") or min_height > 2.0 or \
        max_height < 0.3:
        return False
    return True


def check_closed_receptable(cur_object, object_list):
    id2info_dict = {obj["objectId"]: obj for obj in object_list}
    flag, obj = False, cur_object
    while obj["parentReceptacles"] is not None:
        parent_obj = obj["parentReceptacles"][0]
        parent_obj = id2info_dict[parent_obj]
        if parent_obj['openable'] and not parent_obj['isOpen']:
            flag = True
            break
        else:
            obj = parent_obj
    return flag


def convert_degree_to_direction(angle):
    degree = 0
    if angle >= 315 or angle < 45:
        degree = 0
    elif 45 <= angle < 135:
        degree = 90
    elif 135 <= angle < 225:
        degree = 180
    elif 225 <= angle < 315:
        degree = 270
    return degree


def plot_house_with_room_bounds(reachable_positions, room_list):
    xs = [rp[0] for rp in reachable_positions]
    zs = [rp[1] for rp in reachable_positions]
    xp_list = [[rp[0] for rp in rp_list] for rp_list in room_list]
    zp_list = [[rp[1] for rp in rp_list] for rp_list in room_list]
    fig, ax = plt.subplots()
    ax.scatter(xs, zs, s=100)  # , color='blue'
    for idx, xp, zp in zip(range(len(xp_list)), xp_list, zp_list):
        ax.scatter(xp, zp, s=100 - 25 * idx)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_title("Reachable Positions in the Scene")
    ax.set_aspect("equal")
    plt.show()
    plt.close()


def plot_top_down_frame(controller, room_save_path):
    image = get_top_down_frame(controller)
    image.save(os.path.join(room_save_path, "top_down.png"))


def get_top_down_frame(controller):
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)


def convert_point_list_to_matrix(point_list):
    point_list = [[point["x"], point["y"], point["z"]] for point in point_list]
    return np.array(point_list)

