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
import random, math
from ai2thor.util.metrics import get_shortest_path_to_point


def calculate_path_length(path_list):
    total_length = 0
    for i in range(1, len(path_list)):
        x1, y1 = path_list[i - 1]
        x2, y2 = path_list[i]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_length += distance
    return total_length


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_dir_path", type=str,
                        default="YOUR_PATH")
    parser.add_argument("--pred_path", type=str,
                            default="YOUR_PATH/llama3_1_8B_cap")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=81)
    parser.add_argument("--trial_list", type=str, default="0,5")
    parser.add_argument("--split_path", type=str, help="the path to the split file",
                        default="YOUR_PATH/split_file.json")
    args = parser.parse_args()
    args.trial_list = [int(trial_id) for trial_id in args.trial_list.split(",")]
    full_latex_metric_str = ""
    for split in ["valid", "test"]:
        args.full_pred_path = f"{args.pred_path}/{split}"
        print(args)

        # train/valid/test according to the split file
        # total 4600
        with open(args.split_path) as fin:
            split_json = json.load(fin)
        cur_split = split_json[split]
        cur_split = [idx for idx, _ in cur_split]

        # metric param
        succ_count, total_count = 0, 0
        length_succ_count = 0
        episode_succ_count = 0

        for scene_id in tqdm(cur_split[args.start_idx: args.end_idx],
                             f"testing the {split} split"):
            for trial_id in args.trial_list:
                if not os.path.exists(os.path.join(args.full_pred_path, f"{scene_id}-{trial_id}", "agent_traj.json")):
                    continue
                with open(os.path.join(args.trial_dir_path, f"{scene_id}-{trial_id}", "traj.json")) as fin:
                    gt_data = json.load(fin)
                with open(os.path.join(args.full_pred_path, f"{scene_id}-{trial_id}", "agent_traj.json")) as fin:
                    pred_data = json.load(fin)
                total_count += 1

                succ_count += int(pred_data["task_success"])

                # length weighted
                shortest_traj = pred_data["SPL_shortest"] if "SPL_shortest" in pred_data else pred_data["traj_info"]["corner"]
                # shortest_traj = [(corner["x"], corner["z"]) for corner in shortest_traj]
                shortest_len = calculate_path_length(shortest_traj)
                pred_traj = gt_data["traj_info"]["corner"]
                pred_len = calculate_path_length(pred_traj)
                length_succ_count += shortest_len / max(shortest_len, pred_len) * int(pred_data["task_success"])

                # episode weighed
                shortest_epi_len = len(gt_data["traj_info"]["action"])
                pred_epi_len = len(pred_data["traj_info"]["action"])
                episode_succ_count += shortest_epi_len / max(shortest_epi_len, pred_epi_len) * int(pred_data["task_success"])

        SR = succ_count / total_count * 100
        SPL = length_succ_count / total_count * 100
        SEL = episode_succ_count / total_count * 100

        metric_json_data = {"SR": SR, "SPL": SPL, "SEL": SEL}

        scene_metric_log = os.path.join(args.full_pred_path, "metric.txt")
        with open(scene_metric_log, "w") as fout:
            fout.write(split + "\n")
            fout.write(json.dumps(metric_json_data) + "\n")
            latex_format = f"&{SR:.2f}&{SPL:.2f}&{SEL:.2f}"
            fout.write(json.dumps(latex_format) + "\n")
        full_latex_metric_str += latex_format

        print(succ_count, total_count)
        print(f"{split} {total_count} {latex_format}")
    print(full_latex_metric_str)
