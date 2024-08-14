import json
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import random

from sklearn.metrics import classification_report
from pal_code_exec import compute

task = "gsm8k"
# method = "filtered_rebalanced_feedback_diff"
method = "random"

model = "meta-llama-Llama-2-13b-chat-hf"

infile_dir = f"../fof/logs/{task}/fof_answer_rounds_1_feedback_rounds_3/few_shot/{model}/"
test_example_cnt = 100
initial_codes = []
for idx in range(test_example_cnt):
    with open(os.path.join(infile_dir, f"actor{idx}.jsonl"), 'r') as reader:
        jsonl_items = [json.loads(l) for l in reader]
    initial_code = jsonl_items[1]['content'].replace("```", "").strip() + "\nprint(solution())"
    initial_codes.append(initial_code)
initial_codes = [[c] for c in initial_codes]

with open("data/gsm8k/test_shuf.jsonl", 'r') as reader:
    refs = [json.loads(l)['answer'].split('#### ')[-1] for l in reader]

refs = refs[:test_example_cnt]

scores = compute(initial_codes, refs)
y_true = [s[0] == 1 for s in scores]

y_pred = []
if method == "all":
    pred_file = "all 0 (need improvement)"
    y_pred = [0 for i in range(test_example_cnt)]
elif method == "random":
    pred_file = "33.9% as 1, rest as 0"
    y_pred = [random.random() < 0.339 for i in range(test_example_cnt)]
else:
    pred_file = f"/data/yunxiang/LLaMA-Factory/predictions/{task}_{method}_{model}/generated_predictions.jsonl"
    with open(pred_file, 'r') as reader:
        for l in reader:
            item = json.loads(l)
            y_pred.append("`def solution()`: This solution is correct." in item['predict'])

    assert len(y_true) == len(y_pred), f"{len(y_true)} y_true items != {len(y_pred)} y_pred items count"

print(pred_file)
print(classification_report(y_true, y_pred))    


