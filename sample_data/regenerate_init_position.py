from ai2thor.controller import Controller
import prior, os, json
from utils import get_top_down_frame, plot_house_with_room_bounds
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from utils import get_rooms_polymap, try_find_collision_free_starting_position
from tqdm import tqdm
import argparse


def load_scene_dir(dir_path):
    scene_name_list = os.listdir(dir_path)
    scene_dict = {}
    for scene_name in scene_name_list:
        scene_path = os.path.join(dir_path, scene_name)
        if not os.path.isdir(scene_path):
            continue
        for file in os.listdir(scene_path):
            if file.endswith(".json"):
                scene_path = os.path.join(scene_path, file)
                break
        assert scene_path.endswith(".json")
        scene_id = int(scene_name.split("_")[0])
        with open(scene_path) as fin:
            room_json = json.load(fin)
            assert scene_id not in scene_dict
            scene_dict[scene_id] = (room_json, scene_path)

    return scene_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=250)
    parser.add_argument("--holodeck", type=str, default="../Holodeck/data/scenes_agent")

    args = parser.parse_args()

    holodeck_path = args.holodeck
    scene_dict = load_scene_dir(holodeck_path)
    print("finish loading")

    controller = Controller(action_hook_runner=ProceduralAssetHookRunner(asset_directory=
                                                                         "../Holodeck/data/objaverse_holodeck"
                                                                         "/09_23_combine_scale"
                                                                         "/processed_2023_09_23_combine_scale",
                                                                         verbose=False))
    print("start to run the controller")

    for idx in tqdm(range(args.start_idx, args.end_idx)):
        print(idx)
        if idx not in scene_dict:
            continue
        scene_json, scene_path = scene_dict[idx]
        if "init_flag" in scene_json["metadata"]:
            continue
        controller.reset(scene=scene_json)
        house_poly = get_rooms_polymap(scene_json)
        init_pos = try_find_collision_free_starting_position(scene_json,
                                                             controller,
                                                             house_poly)
        agent_info = scene_json["metadata"]["agent"]
        agent_info["position"]["x"] = init_pos[0]
        agent_info["position"]["z"] = init_pos[1]
        agent_pose = scene_json["metadata"]["agentPoses"]
        agent_pose["arm"]["position"]["x"] = init_pos[0]
        agent_pose["arm"]["position"]["z"] = init_pos[1]
        agent_pose["default"]["position"]["x"] = init_pos[0]
        agent_pose["default"]["position"]["z"] = init_pos[1]
        agent_pose["locobot"]["position"]["x"] = init_pos[0]
        agent_pose["locobot"]["position"]["z"] = init_pos[1]
        agent_pose["stretch"]["position"]["x"] = init_pos[0]
        agent_pose["stretch"]["position"]["z"] = init_pos[1]

        scene_json["metadata"]["init_flag"] = True

        with open(scene_path, "w") as fin:
            json.dump(scene_json, fin, indent=4)

        print(init_pos)
