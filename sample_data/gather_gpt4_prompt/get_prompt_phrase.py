import os
import json
import argparse
from completion_utils import *
from transformers import set_seed
import time
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from openai import OpenAI
from room_constant import room_exemplar_list, room_category, feature_list

room2category = {room: cate for cate, room_list in room_category.items() for room in room_list}
room_type_list = [room for cate, room_list in room_category.items() for room in room_list]

# instruction
# exemplar
# test case
prompt_template = {"instruction": "Create a detailed and fluent description for a room based on the given "
                                  "type and features in two steps. "
                                  "Step 1: provide the value of each feature. "
                                  "Step 2: write a short phrase to describe the room type with the values.",
                   "input": "The given room type is \"{}.\" The feature list is: "
                            "\"{}.\"",
                   "output": "Step 1: {}\n Step 2: {}"
                   }


def build_chat_prompt(room_type, feat_list, aug_room_exemplar_list):
    message_list = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_template["instruction"]},
        {"role": "assistant", "content": "Yes, I understand."}
    ]
    sampled_exemplar_list = random.sample(aug_room_exemplar_list, k=5)
    for cur_e in sampled_exemplar_list:
        cur_feature_list = [(key, value) for key, value in cur_e["feat_dict"].items()]
        feature_str = " ".join([f"({idx}) {f[0]}" for idx, f in enumerate(cur_feature_list, start=1)])
        input_prompt = prompt_template["input"].format(cur_e["room_type"], feature_str)
        message_list.append({"role": "user", "content": input_prompt})
        value_str = " ".join([f"({idx}) {f[1]}" for idx, f in enumerate(cur_feature_list, start=1)])
        message_list.append({"role": "assistant", "content":
            prompt_template["output"].format(value_str, cur_e["prompt"])})

    feature_str = " ".join([f"({idx}) {f}" for idx, f in enumerate(feat_list, start=1)])
    input_prompt = prompt_template["input"].format(room_type, feature_str)

    message_list.append({"role": "user", "content": input_prompt})
    return message_list


def load_previous_data(output_file):
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            output_data = [json.loads(line) for line in f.readlines()]
            # for line in f.readlines():
            #     json.loads(line)
    else:
        output_data = []
    return output_data


def prepare_phrase_token(data_list, tokenizer):
    all_phrase_tokens = []
    for d in data_list:
        phrase = d["prompt"]
        phrase_token = tokenizer.tokenize(phrase)
        all_phrase_tokens.append(phrase_token)
    return all_phrase_tokens


def parse_last_step(text, default_value):
    # remove "steps" for cot
    text = text.lower()
    pattern = r'(step \d+:)'
    text = re.split(pattern, text, flags=re.IGNORECASE)
    text = [t.strip() for t in text if t.strip()]
    # for cot prompting, we need to first check the last step

    answer_idx = -1
    for i in range(len(text)):
        if text[i] == "step 2:":
            answer_idx = i + 1
            break
    if answer_idx == -1:
        return default_value
    ans = text[answer_idx].replace("write a short phrase to describe the room type with the values.", "")
    ans = ans.strip()
    return ans


def parse_feat_dict(text, key_list):
    if text is None:
        return None
    # remove "steps" for cot
    text = text.lower()
    if "step 1:" not in text:
        return None
    text = text[text.index("step 1:"):]
    pattern = r'step \d+:'
    text = re.split(pattern, text, flags=re.IGNORECASE)
    text = [t.strip() for t in text if t.strip()]
    # for cot prompting, we need to first check the last step
    step1_text = text[0]
    if "(1)" not in step1_text:
        return None
    step1_text = step1_text[step1_text.index("(1)"):]
    value_pattern = r'\(\d+\)'
    field_value = re.split(value_pattern, step1_text, flags=re.IGNORECASE)
    field_value = [f.strip() for f in field_value if f.strip()]

    if len(key_list) != len(field_value):
        print(len(key_list), len(field_value))
        return None
    value_dict = {}
    for key, value in zip(key_list, field_value):
        value_dict[key] = value
    return value_dict


SLEEP_SECS = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="../../../new_trajectories/")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10000)
    parser.add_argument("--temp", type=float, default=1.4)
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")  # "gpt-4-1106-preview")
    # gpt-4-turbo-2024-04-09 gpt-4o-2024-05-13 gpt-3.5-turbo-0125
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, "phrase_log.txt"), "a") as fout:
        fout.write(str(args))

    print(args)

    client = OpenAI(
        api_key="",
        max_retries=5,
    )

    if args.model_name in {"gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13", "gpt-4o",
                           "gpt-3.5-turbo-0125", "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-3.5-turbo"}:
        chat_model_flag = True
    else:
        raise ValueError(f"model {args.model_name} is not supported")

    # get labeled in-context example
    print("chat_model_flag", chat_model_flag)
    phrase_list_path = os.path.join(args.output_dir, f"phrase_list.json")  # _{args.start_idx}_{args.end_idx}
    fail_list_path = os.path.join(args.output_dir, f"failed_prompt_list.json")  # _{args.start_idx}_{args.end_idx}

    prev_data = load_previous_data(phrase_list_path)
    aug_room_exemplar_list = room_exemplar_list + [d for d in prev_data if d["feat_dict"] is not None]
    start_idx = len(prev_data) + args.start_idx
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    all_phrase_tokens = prepare_phrase_token(prev_data, scorer._tokenizer)

    # set pandas random seed
    # pandas use numpy random seed by default
    set_seed(start_idx)

    phrase_file = open(phrase_list_path, "a")
    fail_file = open(fail_list_path, "a")

    idx_d = start_idx
    while idx_d < args.end_idx:
        start_time = time.time()
        if idx_d < len(room_type_list):
            labeled_instance = {"id": idx_d,
                                "prompt": room_type_list[idx_d],
                                "room_type": room_type_list[idx_d],
                                "feats": None,
                                "chatgpt": None,
                                "feat_dict": None}
        else:
            # random sample a room type and features
            cur_room_type = random.choice(room_type_list)
            feat_count = random.choices([1, 2, 3], weights=[0.4, 0.5, 0.1])[0]

            selected_feature_index = random.sample(range(len(feature_list)), k=feat_count)
            cur_feat_list = [feature_list[idx] for idx in selected_feature_index]

            message_list = build_chat_prompt(cur_room_type, cur_feat_list, aug_room_exemplar_list)

            response = chat_with_backoff(client,
                                         model=args.model_name, messages=message_list,
                                         max_tokens=256, temperature=args.temp, top_p=1, frequency_penalty=0,
                                         presence_penalty=0, n=1)
            chatgpt_text = response.choices[0].message.content.strip()

            # chatgpt_text = "Step 1: (1) industrial-chic\nStep 2: gym"
            # parse
            room_phrase = parse_last_step(chatgpt_text, default_value=cur_room_type).capitalize()
            feat_dict = parse_feat_dict(chatgpt_text, cur_feat_list)
            labeled_instance = {"id": idx_d,
                                "prompt": room_phrase,
                                "room_type": cur_room_type,
                                "feats": cur_feat_list,
                                "chatgpt": chatgpt_text,
                                "feat_dict": feat_dict}
        cur_phrase_token = scorer._tokenizer.tokenize(labeled_instance["prompt"])
        sim_check_start = time.time()
        with Pool(4) as p:
            rouge_scores = p.map(partial(rouge_scorer._score_lcs, cur_phrase_token),
                                 all_phrase_tokens)
        sim_check_end = time.time()

        rouge_scores = [score.fmeasure for score in rouge_scores]
        # write duplicate case to a new file
        is_duplicate = max(rouge_scores) > 0.8 if rouge_scores else False
        # TODO the phrase that equals room types are wrong
        if is_duplicate or not labeled_instance["prompt"] or labeled_instance["feat_dict"] is None:
            output_file = fail_file
        else:
            output_file = phrase_file
        output_file.write(json.dumps(labeled_instance) + "\n")
        output_file.flush()
        # output log when the new case is not duplicate
        # then, update the idx
        end_time = time.time()
        if is_duplicate:
            print("duplicated", idx_d, end="\t", flush=True)
            print("sim check:", sim_check_end - sim_check_start, end="\t")
        elif not labeled_instance["prompt"]:
            print("empty", idx_d, end="\t", flush=True)
            print("sim check:", sim_check_end - sim_check_start, end="\t")
        elif labeled_instance["feat_dict"] is None:
            print("cannot be parsed", idx_d, end="\t", flush=True)
            print("sim check:", sim_check_end - sim_check_start, end="\t")
        else:
            print("instance", idx_d, end="\t", flush=True)
            print("sim check:", sim_check_end - sim_check_start, end="\t")
            if (idx_d - start_idx + 1) % 3 == 0:
                print()
                print(f"{end_time - start_time} secs used, sleep {max(0.0, SLEEP_SECS - end_time + start_time)} secs")
            idx_d += 1
            all_phrase_tokens.append(cur_phrase_token)
            aug_room_exemplar_list.append(labeled_instance)
        time.sleep(max(0.0, SLEEP_SECS - end_time + start_time))  # keep that 60 examples per min if batch_size = 1
    if start_idx >= args.end_idx:
        print("no unlabeled data")
    phrase_file.close()
    fail_file.close()
