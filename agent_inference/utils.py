import io

import tqdm
import logging
import os
import sys
import datasets
import transformers
from dataclasses import asdict
import shutil
from torch import nn
import torch
from torch.utils.data import Dataset
from copy import deepcopy
from dataclasses import dataclass, field
from sklearn.metrics import precision_recall_curve
import numpy as np
import json
from tqdm import tqdm
from typing import Dict, Optional, Sequence
import re
from PIL import Image

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further "
                    "context. Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:",
    "prompt_no_input": "Below is an instruction that describes a task. "
                       "Write a response that appropriately completes the request.\n\n"
                       "### Instruction:\n{}\n\n### Response:",
}

ICL_PROMPT_DICT = {
    "prompt_input_ins": "Below is an instruction that describes a task, paired with an input that provides further "
                    "context. Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{}\n\n",
    "prompt_input": "### Input:\n{}\n\n### Response:",
}


def init_logger(training_args, log_level):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # init a formatter to add date information
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # init a file handler and a stream handler
    fh = logging.FileHandler(os.path.join(training_args.output_dir, "train.log"), encoding="utf-8", mode="a")
    fh.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    # set formatter to handlers
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add those handlers to the root logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # the logger level of huggingface packages
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.enable_propagation()

    return logger


def format_args(args):
    args_as_dict = asdict(args)
    # args_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in args_as_dict.items()}
    attrs_as_str = [f"{k}={v}," for k, v in sorted(args_as_dict.items())]
    return f"{args.__class__.__name__}\n({' '.join(attrs_as_str)})"


def ds_init_output_dir(training_args):
    if os.path.exists(training_args.output_dir):
        if os.path.exists(os.path.join(training_args.output_dir, "checkpoint_finish")) > 0:
            raise ValueError(
                "training/inference process in dir {} is finished, plz clear it manually.".format(training_args.output_dir))
        if training_args.do_train:
            shutil.rmtree(training_args.output_dir, ignore_errors=True)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    os.system("touch {}".format(os.path.join(training_args.output_dir, "train.log")))


def revise_mnli_models(model_name_or_path, mnli_model, neutral_id, entail_id):
    if "bart" in model_name_or_path:
        head = mnli_model.classification_head
        linear = head.out_proj  # n x 3
    elif "roberta" in model_name_or_path:
        head = mnli_model.classifier
        linear = head.out_proj
    elif "deberta" in model_name_or_path:
        linear = mnli_model.classifier
    else:
        raise ValueError

    # copy weight and bias
    hidden_size = linear.weight.shape[-1]
    new_linear = nn.Linear(hidden_size, 2)  # n x 2
    with torch.no_grad():
        linear_weight = torch.stack([linear.weight[neutral_id, :], linear.weight[entail_id, :]], dim=0)
        linear_bias = torch.stack([linear.bias[neutral_id], linear.bias[entail_id]])
        new_linear.weight.data = linear_weight
        new_linear.bias.new_data_list = linear_bias

    if "bart" in model_name_or_path:
        mnli_model.classification_head.out_proj = new_linear
    elif "roberta" in model_name_or_path:
        mnli_model.classifier.out_proj = new_linear
    elif "deberta" in model_name_or_path:
        mnli_model.classifier = new_linear

    # change config
    mnli_model.config.num_labels = 2

    if hasattr(mnli_model, "num_labels"):
        mnli_model.num_labels = 2

    mnli_model.eval()

    return mnli_model


def is_main_process(local_rank):
    return local_rank == 0 or local_rank == -1


def average_precision_score(y_true, y_score, pos_label=1):
    precision, recall, _ = precision_recall_curve(
        y_true, y_score, pos_label=pos_label
    )
    print(len(precision), precision)
    print(len(recall), recall)
    recall_diff, precision = np.diff(recall), np.array(precision)[:-1]
    high_precision_mask = precision > 0.5
    print(len(high_precision_mask), high_precision_mask)
    recall_diff, precision = recall_diff[high_precision_mask], precision[high_precision_mask]
    # print(len(recall_diff), recall_diff)
    # print(len(precision), precision)
    return -np.sum(recall_diff * precision)


def store_generation(training_args, text_list, split_name):
    with open(os.path.join(training_args.output_dir, "{}.jsonl".format(split_name)), "w") as fout:
        for ri, rp, tp, i, l, p in tqdm(zip(*text_list), "output generations"):
            fout.write(json.dumps({"input": i, "label": l, "pred": p,
                                   "raw_input": ri, "raw_pred": rp, "text_pred": tp}) + "\n")


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.new_data_list
        output_embeddings = model.get_output_embeddings().weight.new_data_list

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_list, processor, max_length, image_dir):
        super(SupervisedDataset, self).__init__()
        logging.info("Loading data...")
        self.processor = processor
        self.max_length = max_length
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        self.tokenized_text = self.tokenize_function(data_list)
        self.image_dir = image_dir
        self.data_list = data_list
        logging.info("Formatting inputs...")

    def tokenize_function(self, example_list):
        texts = []
        for example in example_list:
            question = example["query"]

            result = [seg.strip() for seg in re.split(r'(<image>)', question) if seg.strip()]
            content_list = [{"type": "image"} if seg == "<image>"
                            else {"type": "text", "text": seg} for seg in result]

            messages = [
                {
                    "role": "user",
                    "content": content_list
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example["answer"].strip()}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())

        batch = self.processor(text=texts, padding='do_not_pad',
                               max_length=self.max_length, truncation = True)

        labels = batch["input_ids"].clone()
        # mask the input
        for i in range(len(labels)):
            cur_label = torch.tensor(labels[i], dtype=torch.int64)
            cur_label[cur_label == self.processor.tokenizer.pad_token_id] = -100
            cur_label[cur_label == self.image_token_id] = -100
            eot_position = (cur_label == 32002).nonzero(as_tuple=False)
            assert len(eot_position) == 2
            cur_label[i, :eot_position[0] + 6] = -100
            labels[i] = cur_label

            # update input_ids and attention_mask
            batch["input_ids"][i] = torch.tensor(batch["input_ids"][i], dtype=torch.int64)
            batch["attention_mask"][i] = torch.tensor(batch["attention_mask"][i], dtype=torch.int64)

        batch["labels"] = labels

        return batch

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return_dict = {"input_ids": self.tokenized_text["input_ids"][i],
                       "attention_mask": self.tokenized_text["attention_mask"][i]}

        image_list = [Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
                      for image_path in self.data_list[i]["image"]]
        return_dict["image"] = image_list
        return return_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, instance_list: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instance_list]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        attention_mask = [instance["attention_mask"] for instance in instance_list]
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0)
        element_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        # reset [PAD] token by -100, so GPT2LMHealModel will not compute loss on that
        # but not the first [PAD] token, which is [EOS]
        labels = [instance["labels"] for instance in instance_list]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        element_dict["labels"] = labels
        # pad image
        image_list = [instance["image"] for instance in instance_list]
        image_dict = self.processor(images=image_list, return_tensors="pt", padding=True)
        element_dict.update(image_dict)

        return element_dict


def json_load(f, mode="r"):
    """Load a .json file into a dictionary."""
    with open(f, mode=mode) as fin:
        list_data_dict = [json.loads(line) for line in fin]
    return list_data_dict


def load_json_dial(f, image_base_url, mode="r", debug=False):
    with open(f, mode=mode) as fin:
        list_data_dict = [json.loads(line) for line in fin]
    if debug:
        list_data_dict = list_data_dict[:1000]

    data_list = []
    for data in tqdm(list_data_dict, "reformat data"):
        image_list = [os.path.join(image_base_url, image_path) for image_path in data["image"]]
        data_list.append({"id": data["id"], "image": image_list,
                          "query": data["conversations"][0]["value"],
                          "answer": data["conversations"][1]["value"]})
    return data_list

word_map = {"yes": 1, "no": 0}
def parse_label(text):
    text = [line.strip() for line in text.lower().split("\n") if line.strip()]
    text = "\n".join(text[::-1])
    if text in word_map:
        return word_map[text]
    index_map = {"yes": 1e5, "no": 1e5}
    for word in word_map:
        if word in text:
            index_map[word] = text.index(word)
    if index_map["yes"] == 1e5 and index_map["no"] == 1e5:
        return -1
    elif index_map["yes"] < index_map["no"]:
        return 1
    elif index_map["yes"] > index_map["no"]:
        return 0
    else:
        return -1


class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, example_list):
        texts = []
        images = []
        for example in example_list:
            image_list = example["image"]
            question = example["query"]

            image_list = [Image.open(image_path).convert('RGB') for image_path in image_list]
            result = [seg.strip() for seg in re.split(r'(<image>)', question) if seg.strip()]
            content_list = [{"type": "image"} if seg == "<image>"
                            else {"type": "text", "text": seg} for seg in result]

            messages = [
                {
                    "role": "user",
                    "content": content_list
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example["answer"].strip()}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append(image_list)

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        # mask the input
        for i in range(labels.shape[0]):
            eot_position = (labels[i] == 32002).nonzero(as_tuple=False)
            assert len(eot_position) == 2
            labels[i, :eot_position[0] + 6] = -100
        eot_position = (labels == 32002).nonzero(as_tuple=True)

        batch["labels"] = labels

        return batch
