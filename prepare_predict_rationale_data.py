import json
import argparse

from util import *


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str)
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-chat-hf")
parser.add_argument("--split", type=str, choices=["train", "dev", "test", "train_for_inference"])
parser.add_argument("--chunk", type=int, default=1)
args = parser.parse_args()

task = args.task
model = args.model.replace("/", '-')
split = args.split
chunk = 1 if split == "train" else args.chunk

if split == "test":
    with open(f"data/{task}/test_shuf.jsonl", 'r') as reader:
        items = [json.loads(l) for l in reader]
elif split == "train" or split == "dev" or split == "train_for_inference":
    with open(f"logs/{task}/generations_{model}_score.jsonl", 'r') as reader:
        items = [json.loads(l) for l in reader]
    # assert len(items) == 7473
    dev_size = TASK2DEV_SIZE[task]
    if split == "train" or split == "train_for_inference":
        items = items[:dev_size]
    else:
        items = items[dev_size:]

instr = TASK2ACTOR_USER_INITIAL_PROMPT[task].replace(" End your response with [END].\n", "")
rationale_items = []
for item in items:
    if split != "train": # test
        rationale_item = {
            "instruction" : instr.replace("{question}", item['question']),
            "input" : "",
            "output" : f"\n```\n\n```\n\n### END ###",
            }
        rationale_items.append(rationale_item)
    else:
        for s, g in zip(item['score'], item['generated_answers']):
            if s == 1:
                g = clean_cot(g)
                rationale_item = {
                    "instruction" : instr.replace("{question}", item['question']),
                    "input" : "",
                    "output" : g.strip(),
                    }
                rationale_items.append(rationale_item)

if split == "train":
    print("pre-deduplication train cnt:", len(rationale_items))
    rationale_items = [dict(t) for t in {tuple(d.items()) for d in rationale_items}]
    print("post-deduplication train cnt:", len(rationale_items))

outfile = f"../LLaMA-Factory/data/{task}_{split}_rationale_{model}.json"
save_to_file(rationale_items, outfile, chunk)
