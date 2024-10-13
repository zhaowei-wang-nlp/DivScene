import argparse
import json, os
from transformers import set_seed
from utils import get_llava_format_data_no_diff, train_data_filter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, help="input trajectory dir",
                        default="../../new_trajectories/holodeck_4614")
    parser.add_argument("--output_path", type=str, help="input trajectory dir",
                        default="../../new_trajectories/train_file_4614")
    parser.add_argument("--sample_rate", type=int, help="the rate to downsample MoveAhead",
                        default=4)
    parser.add_argument("--trial_num", type=int, help="the number of trials to use in the sampling",
                        default=5)
    parser.add_argument("--seed", type=int, help="the random seed of generating random numbers",
                        default=42)
    parser.add_argument("--image_num", type=int, help="the number of images",
                        default=4)
    parser.add_argument("--save_eval_data", action="store_true", help="store evaluation data")
    parser.add_argument("--action_his_num", type=int, default=16, help="whether use full action history")
    parser.add_argument("--debug_data", action="store_true", help="store debug version of the data")
    parser.add_argument("--use_cot", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)
    args.split_path = os.path.join(args.dir_path, "split_file.json")
    print(args)
    prev_image_num = args.image_num - 1

    # train/valid/test according to the split file
    # total 4600
    with open(args.split_path) as fin:
        split_json = json.load(fin)
    split_json["train"] = set(idx for idx, _ in split_json["train"])
    split_json["valid"] = set(idx for idx, _ in split_json["valid"])
    split_json["test"] = set(idx for idx, _ in split_json["test"])

    task_list = sorted(os.listdir(args.dir_path))
    task_dict = {"train": [], "valid": [], "test": []}
    total_task_count = 0
    for task_name in task_list:
        task_path = os.path.join(args.dir_path, task_name)
        if not os.path.isdir(task_path):
            continue

        scene_id, trial_id = task_name.split("-")
        scene_id, trial_id = int(scene_id), int(trial_id)
        if scene_id in split_json["train"]:
            cur_split = "train"
        elif scene_id in split_json["valid"]:
            cur_split = "valid"
        elif scene_id in split_json["test"]:
            cur_split = "test"
        else:
            raise KeyError("Wrong scene id")
        cur_split_list = task_dict[cur_split]
        cur_split_list.append(task_name)
        total_task_count += 1
    print(total_task_count, len(task_dict["train"]), len(task_dict["valid"]), len(task_dict["test"]))
    train_split_data = get_llava_format_data_no_diff(args.dir_path, task_dict["train"], args.trial_num,
                                             args.sample_rate, args.image_num, args.action_his_num, args.use_cot)
    train_split_data = train_data_filter(train_split_data, args.dir_path)

    print(f"There are {len(train_split_data)} examples in total.")
    if args.action_his_num is None:
        full_flag = ""
    elif args.action_his_num > 0:
        full_flag = f"stp{args.action_his_num}_"
    elif args.action_his_num == 0:
        full_flag = "full_"
    else:
        raise KeyError("Wrong action_his_num")
    if args.use_cot:
        prefix = "new_cot_nd"
    else:
        prefix = "new_holodeck_nd"

    output_path = os.path.join(args.output_path, f"{prefix}_{full_flag}train_tn{args.trial_num}"
                                                 f"_sr{1 / args.sample_rate:.2f}_in{args.image_num}.json")
    # save train
    with open(output_path, "w") as fout:
        json.dump(train_split_data, fout, indent=4)

    # save debug version
    if args.debug_data:
        output_path = os.path.join(args.output_path,
                                   f"{prefix}_debug_{full_flag}train_tn{args.trial_num}_sr{args.sample_rate}_in{args.image_num}.json")
        with open(output_path, "w") as fout:
            json.dump(train_split_data[:5000], fout, indent=4)

    if args.save_eval_data:
        valid_split_data = get_llava_format_data_no_diff(args.dir_path, task_dict["valid"], 1, 1, args.image_num,
                                                 args.action_his_num, args.use_cot)
        test_split_data = get_llava_format_data_no_diff(args.dir_path, task_dict["test"], 1, 1, args.image_num,
                                                args.action_his_num, args.use_cot)
        print(f"There are {len(valid_split_data)}/{len(test_split_data)} examples in total.")
        # save valid
        with open(os.path.join(args.output_path, f"{prefix}_{full_flag}valid_in{args.image_num}.json"), "w") as fout:
            json.dump(valid_split_data, fout, indent=4)
        # save test
        with open(os.path.join(args.output_path, f"{prefix}_{full_flag}test_in{args.image_num}.json"), "w") as fout:
            json.dump(test_split_data, fout, indent=4)
