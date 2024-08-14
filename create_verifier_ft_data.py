import json
import os
import argparse

from util import clean_cot, TASK2VERIFY_FT_PROMPT, TASK2DEV_SIZE, TASK_LIST

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str, choices=TASK_LIST)
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-chat-hf")
parser.add_argument("--split", type=str, default="train", choices=["train", "dev", "test", "train_cot_ft", "dev_cot_ft", "test_cot_ft", "dev_refinement", "test_refinement"])
parser.add_argument("--greedy", action='store_true')

args = parser.parse_args()
TASK = args.task
split = args.split
deployment_id = args.model.replace('/', '-')
use_greedy_decoding = args.greedy
dev_size = TASK2DEV_SIZE[TASK]

infile = f"logs/{TASK}/generations_{deployment_id}_score.jsonl"
if split == "test":
    infile = infile.replace(".jsonl", "_test.jsonl")
elif split == "train_cot_ft":
    infile = f"logs/{TASK}/cot_ft_generations_{deployment_id}_score.jsonl"
elif split == "dev_cot_ft":
    infile = f"logs/{TASK}/cot_ft_generations_{deployment_id}_score_dev.jsonl"
elif split == "test_cot_ft":
    infile = f"logs/{TASK}/cot_ft_generations_{deployment_id}_score_test.jsonl"
elif split == "dev_refinement":
    infile = f"logs/{TASK}/refinement_generations_{deployment_id}_score_dev.jsonl"
elif split == "test_refinement":
    infile = f"logs/{TASK}/refinement_generations_{deployment_id}_score_test.jsonl"
    

    # infile_cot = "../LLaMA-Factory/predictions/gsm8k_nl_dev_rationale_train_rationale_meta-llama-Llama-2-13b-chat-hf_lr1e-5_step_1548/generated_predictions.jsonl.00"
    # with open(infile_cot, 'r') as reader:
    #    pred_items = [json.loads(l) for l in reader]
if use_greedy_decoding:
    infile += "_greedy"

with open(infile, 'r') as reader:
    items = [json.loads(l) for l in reader]

if split == "train":
    items = items[:dev_size]
elif split == "dev":
    if len(items) > dev_size:
        items = items[dev_size:]

# if split == "dev_cot_ft":
#     assert len(pred_items) == len(items)
# else:
#     pred_items = items
VERIFY_FT_PROMPT = TASK2VERIFY_FT_PROMPT[TASK]
data = []
for item in items:
    question = item['question']
    for g, score in zip(item['generated_answers'], item['score']):
        text = VERIFY_FT_PROMPT.replace("{question}", question).replace("{solution}", clean_cot(g))
        # text = f"Question: {question}\n\nSolution:\n{clean_cot(g)}\n\nIs this solution correct or not?"
        label = score
        data.append((text, label))

print(f"{len(data)} in total")
if "train" in split:
    # deduplication
    data = set(data)
    print(f"{len(data)} in total after deduplication")
data = [{"text" : text, "target" : label} for text, label in data]

outdir = f"data/{TASK}/verifier/{deployment_id}/"
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, f"{split}.json")
if not use_greedy_decoding:
    outfile += "_sample"
with open(outfile, 'w') as writer:
    for d in data:
        writer.write(json.dumps(d)+'\n')


print(f"writing to {outfile}")
