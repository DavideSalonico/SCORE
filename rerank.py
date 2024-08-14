import json
import argparse
import numpy as np
from tqdm import tqdm
from util import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str, choices=TASK_LIST)
parser.add_argument("--model", type=str, choices=MODEL_LIST)
parser.add_argument("--split", type=str, choices=["dev", "test"])
parser.add_argument("--vtype", type=str, choices=['verifier', 'voting_verifier'], default='voting_verifier')
parser.add_argument("--conf", type=float)
args = parser.parse_args()

TASK = args.task
MODEL = args.model.replace('/', '-')
# MODEL = "google-gemma-7b-it"
BEST_DEV_CONFIDENCE = args.conf
# VEIFIER_TYPE = "verifier"
VERIFIER_TYPE = args.vtype

# BEST_DEV_CONFIDENCE = 0.74
SPLIT = args.split
print(TASK, MODEL, VERIFIER_TYPE)
if SPLIT == "test":
    with open(f"data/{TASK}/{SPLIT}_shuf.jsonl", 'r') as reader:
        data_items = [json.loads(l) for l in reader]
        QUESTION_CNT = len(data_items)
elif SPLIT == "dev":
    with open(f"data/{TASK}/train_shuf.jsonl", 'r') as reader:
        data_items = [json.loads(l) for l in reader]
    QUESTION_CNT = len(data_items) - TASK2DEV_SIZE[TASK]

SAMPLE_CNT = 10

def all_same(items):
    return len(set(items)) == 1

def read_vitems(infile):
    items = [[] for i in range(QUESTION_CNT)]
    with open(infile, 'r') as reader:
        for idx, l in enumerate(reader):
            item = json.loads(l)
            if item['predict'] == 1:
                item['confidence'] = 1 - item['confidence']
            items[idx//SAMPLE_CNT].append(item['confidence'])
    return items

vscore_init = read_vitems(f"checkpoint/verifier/{TASK}/{MODEL}/{SPLIT}.json_sample")
vscore_refine = read_vitems(f"checkpoint/verifier/{TASK}/{MODEL}/{SPLIT}_refinement.json")

def read_aitems(infile):
    items = []
    with open(infile, 'r') as reader:
        for l in reader:
            item = json.loads(l)
            items.append(list(zip(item['score'], item['extracted_answers'])))
    return items

ascore_refine = read_aitems(f"logs/{TASK}/refinement_generations_{MODEL}_score_{SPLIT}.jsonl_greedy")
if SPLIT == "test":
    ascore_init = read_aitems(f"logs/{TASK}/generations_{MODEL}_score_{SPLIT}.jsonl")
elif SPLIT == "dev":
    ascore_init = read_aitems(f"logs/{TASK}/generations_{MODEL}_score.jsonl")
    dev_size = TASK2DEV_SIZE[TASK]
    ascore_init = ascore_init[dev_size:]
"""
key = 'score' if VERIFIER_TYPE == "verifier" else "extracted_answers"
with open(f"logs/{TASK}/refinement_generations_{MODEL}_score_test.jsonl", 'r') as reader:
    ascore_refine = [json.loads(l)[key] for l in reader]
with open(f"logs/{TASK}/generations_{MODEL}_score_test.jsonl", 'r') as reader:
    ascore_init = [json.loads(l)[key] for l in reader]
"""
assert all_same([len(vscore_init), len(vscore_refine), len(ascore_refine), len(ascore_init)]), f"len(vscore_init) = {len(vscore_init)}, len(vscore_refine) = {len(vscore_refine)}, len(ascore_refine) = {len(ascore_refine)}, len(ascore_init) = {len(ascore_init)}"

def rerank_accu(v_a_score):
    """
    v_a_score = [[(v1, (a1, x1)), ...], ...]
    """
    # rerank and select top-1
    if VERIFIER_TYPE == "verifier":
        ascores = [sorted(item, key=lambda x:x[0])[0][1][0] for item in v_a_score]
    elif VERIFIER_TYPE == "voting_verifier":
        ascores = []
        for item in v_a_score:
            probability_dict = {}
            for probability, ax in item:
                if ax in probability_dict:
                    probability_dict[ax] += probability
                else:
                    probability_dict[ax] = probability
            
            highest_probability_ax = max(probability_dict, key=probability_dict.get)
            ascores.append(highest_probability_ax[0])
    else:
        raise ValueError("invalid VERIFIER_TYPE")
    # select the answer with most probability mass

    return 100*sum(ascores)/len(ascores)

print("sample and rerank")
v_a_score = [[(v, a) for v, a in zip(vscore, ascore)] for vscore, ascore in zip(vscore_init, ascore_init)]
print(f"accu: {rerank_accu(v_a_score):.2f}")

print("sample, self-correct and rerank")
if BEST_DEV_CONFIDENCE is None:
    all_confidence = [round(i, 3) for i in np.linspace(0.5,1,501).tolist()]
else:
    all_confidence = [BEST_DEV_CONFIDENCE]
best_accu, best_dev_conf = 0, None
for BEST_DEV_CONFIDENCE in tqdm(all_confidence):
    v_a_score = []
    for vscore_init_item, ascore_init_item, vscore_refine_item, ascore_refine_item in zip(vscore_init, ascore_init, vscore_refine, ascore_refine):
        v_a_score_item = []
        for idx in range(SAMPLE_CNT):
            # v_a_score_item.append((vscore_refine_item[idx], ascore_refine_item[idx]))
            # v_a_score_item.append((vscore_init_item[idx], ascore_init_item[idx]))
            if vscore_init_item[idx] > BEST_DEV_CONFIDENCE:
                # decide to self-correct
                v_a_score_item.append((vscore_refine_item[idx], ascore_refine_item[idx]))
            else:
                v_a_score_item.append((vscore_init_item[idx], ascore_init_item[idx]))
        v_a_score.append(v_a_score_item)
    accu = rerank_accu(v_a_score)
    if accu > best_accu:
        best_accu = accu
        best_dev_conf = BEST_DEV_CONFIDENCE
print(f"accu: {best_accu:.2f} best_dev_confidence: {best_dev_conf}")
