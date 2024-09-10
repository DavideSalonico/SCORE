import re
import json
import random
from datasets import load_dataset

def parse_answer(input_str):
    pattern = r'\\boxed\{([0-9.,]*)\}'
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    return solution


typelists = ["Algebra", "Counting & Probability", "Prealgebra", "Number Theory"]
d = load_dataset("lighteval/MATH")

items = [item for split in ["train", "test"] for item in d[split] if item['level'] == "Level 1" and '$' not in item['problem'] and item['type'] in typelists and parse_answer(item['solution']) is not None and '$' not in item['problem']]
#random.shuffle(items)

print(len(items), "after filtering")
#split2items = {"test":items[:150], "train":items[150:]}
split2items = {"test":items, "train":items[150:]}
for split in ["test", "train"]:
    with open(f"{split}_shuf.jsonl", 'w') as writer:
        for item in split2items[split]:
            item['question'] = item['problem']
            answer = parse_answer(item['solution'])
            assert answer is not None
            item['answer'] = f"{item['solution']}\n### {answer}"
            del item['problem']
            del item['solution']
            writer.write(json.dumps(item)+'\n')
