# -*- coding: utf-8 -*-

import os
import re
import json
import torch
import pickle
import argparse
from fastapi import FastAPI
from pydantic import BaseModel, conbytes
import base64
from PIL import Image
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
import uvicorn


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--model_name_or_path", type=str,
                    default="")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--full_path", type=int, default=1)
args = parser.parse_args()
full_model_path = "YOUR_PATH/{}"
if args.full_path:
    args.model_name_or_path = full_model_path.format(args.model_name_or_path)

print(args)

print("Current loaded model:", args.model_name_or_path.split("/")[-2])

processor = AutoProcessor.from_pretrained(
    args.model_name_or_path,
    do_image_splitting=False
)
processor.image_processor.size['longest_edge'] = 378
processor.image_processor.size['shortest_edge'] = 378

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

            result = [seg.strip() for seg in re.split(r'(<image>)', question) if seg.strip()]
            content_list = [{"type": "image"} if seg == "<image>"
                            else {"type": "text", "text": seg} for seg in result]

            messages = [
                {
                    "role": "user",
                    "content": content_list
                },
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(text.strip())
            images.append(image_list)

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        return batch

model = Idefics2ForConditionalGeneration.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",  # Only available on A100 or H100
).to(args.device)


data_collator = MyDataCollator(processor)


action_set = {"MoveAhead", "RotateRight",
            "RotateLeft", "LookUp", "Done"}
a2i = {"moveahead": 0, "rotateright": 1,
           "rotateleft": 2, "lookup": 3, "done": 4}

def map_action_to_id(action_list):
    id_list = [a2i.get(action.lower(), -1) for action in action_list]
    return id_list


def parse_action(output_str):
    output_str = output_str.strip()
    if output_str.startswith("ASSISTANT:"):
        output_str = output_str[len("ASSISTANT:"):].strip()
    try:
        output_str = output_str[output_str.index("3)") + 2:].strip()
    except Exception as e:
        pass
    output_str = output_str.split()
    for seg in output_str:
        if seg in action_set:
            return seg
    for seg in output_str:
        for action in action_set:
            if action in seg:
                return action
    return "None"

# input structure
class InputData(BaseModel):
    id: str
    query: str
    image: str

class OutputPrediction(BaseModel):
    generated_action: str
    generated_text: str


app = FastAPI()
@app.post("/predict")
def predict(example: InputData):
    example = example.dict()
    image_list_bin = base64.b64decode(example["image"])
    image_list = pickle.loads(image_list_bin)
    example["image"] = image_list
    # example.image = [str(type(img)) for img in example.image]
    batch = data_collator([example])
    batch = {k: v.to(args.device) for k, v in batch.items()}
    with torch.no_grad():
        generated_ids = model.generate(**batch, max_new_tokens=256, min_new_tokens=3,
                                       tokenizer=processor.tokenizer, stop_strings=["<end_of_utterance>"])
    generated_text = processor.batch_decode(generated_ids[:, batch["input_ids"].size(1):], skip_special_tokens=True)
    generated_text = generated_text[0]
    # generated_text = "\nAssistant: MoveAhead  MoveAhead "
    generated_action = parse_action(generated_text)
    return {"action": generated_action, "text": generated_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)