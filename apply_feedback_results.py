import json
import os
import argparse
from tqdm import tqdm

from util import *
from pal_code_exec import compute
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str, choices=TASK_LIST)
parser.add_argument("--model", type=str, choices=MODEL_LIST)
parser.add_argument("--cot_ft", action='store_true')
args = parser.parse_args()
TASK = args.task
model = args.model.replace('/', '-') 
use_cot_ft = args.cot_ft

TASK2RANGE = {
        # "gsm8k_nl": range(0, 16000, 1000),
        # "gsm8k_nl": range(0, 70000, 5000),
        "gsm8k_nl": range(0, 60000, 5000),
        }
TASK2INTERVAL = {
        "gsm8k_nl": 5000,
        "csqa" : 5000,
        }

extract_func = TASK2EXTRACTING_ALL_FUNC[TASK]

def is_in(pred, ref):
    all_numbers = extract_func(pred)
    if len(all_numbers) == 0:
        print("pred no numbers:", pred)
        return -1
    for n in all_numbers:
        if is_same(n, ref):
            return 1
    return 0

def extract_boxed_answer_step(pred):
    lines = [l for l in pred.split('\n') if l != ""]
    # too specific
    for line in lines:
        if "boxed" in line:
            return line
    return lines[-1]


all_items = {'diff':[]}
for method in all_items:
    # for start_idx in TASK2RANGE[TASK]:
    start_idx = 0
    while True:
        corrections_file = f"logs/{TASK}/corrections_{method}_{model}_{start_idx}_{start_idx+TASK2INTERVAL[TASK]}.jsonl_greedy"
        if not os.path.exists(corrections_file):
            break
        with open(corrections_file, 'r') as reader:
            items = [json.loads(l) for l in reader]
        for item in items:
            for g_idx, g in enumerate(item['corrections']):
                if TASK == "gsm8k":
                    item['corrections'][g_idx] = g.replace("```", "").replace("### END ###", "").replace("### END", "").replace('\n\n', '') + '\nprint(solution())'
                elif TASK in ["gsm8k_nl", "csqa"]:
                    item['corrections'][g_idx] = clean_cot(g) 
        all_items[method].extend(items)
        start_idx += TASK2INTERVAL[TASK]

"""
print("build wrong_code_to_direct_corrections index ...")
wrong_code_to_direct_item = dict()
for item in tqdm(all_items['direct']):
    wrong_code_to_direct_item[item['wrong_code']] = item

print("align diff to direct solutions ...")
all_items['direct_clean'] = []
for diff_item in tqdm(all_items['diff']):
    direct_item = wrong_code_to_direct_item[diff_item['wrong_code']]
    all_items['direct_clean'].append(direct_item)
"""

"""
for method in ['diff', 'direct_clean', 'direct']:
    print(all_items[method][:1])
"""

for method in ['diff']:
    # all_items[method] = all_items[method][:20]
    refs = [item['reference'] for item in all_items[method]]
    predictions = [item['corrections'] for item in all_items[method]]
    # 对于 gsm8k_nl_nl, 只要有正确答案出现即可,例如Step 8: The answer is \\\\boxed{20} dollars. The correct answer is \\\\boxed{20} dollars, not \\boxed{24}.
    if TASK == "gsm8k":
        scores = compute(predictions, refs)
    elif TASK == "gsm8k_nl":
        scores = [[is_in(extract_boxed_answer_step(pred), ref) for pred in preds] for preds, ref in zip(predictions, refs)] # only keep final step
    elif TASK == "csqa":
        scores = [[is_in(pred, ref) for pred in preds] for preds, ref in zip(predictions, refs)]


    if use_cot_ft:
        outfile = f"logs/{TASK}/corrections_cot_ft_{method}_{model}_scores.jsonl"
    else:
        outfile = f"logs/{TASK}/corrections_{method}_{model}_scores.jsonl"

    print("writing to ", outfile)
    with open(outfile, 'w') as writer:
        for item, s in zip(all_items[method], scores):
            item['scores'] = s
            writer.write(json.dumps(item)+'\n')

    for TOP_K in [1,3,5]:
        # predictions_at_k = [preds[:TOP_K] for preds in predictions]
        scores_at_k = [s[:TOP_K] for s in scores]
        correct = len([s for s in scores_at_k if 1 in s])
        accu = correct * 100 / len(scores_at_k)
        print(method, f"accu@{TOP_K}", accu)
