import json

with open("csqa_problems", 'r') as reader:
    shots = json.load(reader)['prompt']

outstr = ""
for shot in shots:
    outstr += shot['question'].replace("A:", "Explain your reasoning step by step. Your final answer should be a single letter from A to E, in the form (answer). End your response with [END].")
    outstr += "\n==="
    idx = 1
    for step in shot['rationale'].split('.'):
        if step.strip() != "":
            outstr += f"\nStep {idx}: {step.strip()}."
            idx += 1
    outstr += f"\nStep {idx}: So the answer is ({shot['pred_ans']}). [END]\n===\n"


with open("init.txt", 'w') as writer:
    writer.write(outstr)
    

