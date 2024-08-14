import json
import re
import os
import argparse
from pprint import pprint

from util import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str, choices=TASK_LIST)
parser.add_argument("--model", type=str, choices=MODEL_LIST, default="meta-llama/Llama-2-13b-chat-hf")
parser.add_argument("--greedy", action='store_true')
parser.add_argument("--cot_ft", action='store_true') # use sampled solutions from cot-finetuned (RFT) model instead of the base model

args = parser.parse_args()
use_greedy_decoding = args.greedy
use_cot_ft = args.cot_ft

TASK = args.task
RANGE = TASK2RANGE[TASK]
INTERVAL = TASK2INTERVAL[TASK]
"""
if "gsm8k" in TASK:
    RANGE = range(0, 6500, 500) # only training set
elif TASK == "csqa":
    RANGE = range(0, 8500, 500)
elif TASK == "ld":
    RANGE = 
"""
deployment_id = args.model.replace('/', '-')
extract_all_func = TASK2EXTRACTING_ALL_FUNC[TASK]

items = []
for start_idx in RANGE:
    if use_cot_ft:
        infile = f"logs/{TASK}/feedbacks_more_cot_ft_diff_{deployment_id}_{start_idx}_{start_idx+INTERVAL}.jsonl"
    else:
        infile = f"logs/{TASK}/feedbacks_diff_{deployment_id}_{start_idx}_{start_idx+INTERVAL}.jsonl"
    if use_greedy_decoding:
        infile += "_greedy"
    if not os.path.exists(infile):
        print(f"skipping non-existig infile {infile}")
        continue
    with open(infile, 'r') as reader:
        items.extend([json.loads(l) for l in reader])

# filter error types
empty_step_feedback_pairs = 0
mismatched_step_feedback = 0
copy_failed = 0
final_step_feedback_correctness_failed = 0
wrong_step_might_contain_correct_answer = 0
after_prefilter_step_feedback_pairs_cnt = 0

for item in items:
    if "gsm8k" in TASK: 
        ref = item['answer'].split('#### ')[-1]
    elif TASK in ["csqa", "ld"]:
        ref = item['answer']
    prefilterted_step_feedbacks = []
    for feedback in item['feedbacks']:
        # parse feedback into (step, feedback) pairs
        # lines = [line for line in feedback.split('\n') if line.strip() != ""]
        feedback = clean_cot(feedback)
        lines = [line.strip() for line in re.split("(Step \d+\:|Feedback\:)", feedback) if line.strip() != ""]
        # pprint(lines)
        step_feedback_pairs = []
        for idx, line in enumerate(lines):
            if re.match("^Step \d+\:", line) and idx + 3 < len(lines):
                wrong_step = f"{line} " + lines[idx+1]
                if lines[idx+2] == "Feedback:":
                    gen_feedback = "Feedback: " + lines[idx+3]
                    gen_feedback = gen_feedback.replace("Answer 1", "incorrect answer").replace("Answer 2", "correct answer")
                    step_feedback_pairs.append((wrong_step, gen_feedback))
        # pprint(step_feedback_pairs)
        # exit(0)
        if len(step_feedback_pairs) == 0:
            print("empty_step_feedback_pairs:")
            pprint(feedback)
            print("==="*10)
            empty_step_feedback_pairs += 1
            continue

        # criteria 1: copy incorrect answer
        wrong_answer = clean_cot(item['wrong_code'])
        wrong_lines = [line for line in re.split("Step \d+\:", wrong_answer) if line.strip() != ""]
        wrong_steps = [f"Step {idx+1}:" + line for idx, line in enumerate(wrong_lines)]
        """
        if len(step_feedback_pairs) != len(wrong_steps):
            print("step_feedback_pairs")
            pprint(step_feedback_pairs)
            print("wrong_steps")
            pprint(wrong_steps)
            exit(0)
            continue
        """
        if len(step_feedback_pairs) > len(wrong_steps):
            """
            # this is good
            print("unequal less len(wrong_steps) and len(step_feedback_pairs)!")
            pprint(feedback)
            print("step_feedback_pairs:", step_feedback_pairs)
            print("wrong_steps:", wrong_steps)
            print("==="*10)
            """
            step_feedback_pairs = step_feedback_pairs[:len(wrong_steps)]
        elif len(step_feedback_pairs) < len(wrong_steps):
            # this often results in weired parsing errors
            print("mismatched_step_feedback, more len(wrong_steps) than len(step_feedback_pairs)!")
            pprint(feedback)
            print("step_feedback_pairs:", step_feedback_pairs)
            print("wrong_steps:", wrong_steps)
            print("==="*10)
            mismatched_step_feedback += 1
            continue


        is_copy_failed = False
        for wrong_step, (gen_step, gen_feedback) in zip(wrong_steps, step_feedback_pairs):
            wrong_step_no = re.findall("Step (\d+):", wrong_step)
            gen_step_no = re.findall("Step (\d+):", gen_step)
            if len(wrong_step_no) == 0 or len(gen_step_no) == 0 or wrong_step_no[0] != gen_step_no[0]:
                # step no != 0
                print("copy_failed!")
                print("question:", item['question'])
                print("answer:", item['answer'])
                print("wrong_step:", wrong_step)
                print("  gen_step:", gen_step)
                print("wrong_steps:")
                pprint(wrong_steps)
                print("step_feedback_pairs:")
                pprint(step_feedback_pairs)
                print("==="*10)
                is_copy_failed = True
                break
        if is_copy_failed:
            copy_failed += 1
            continue

        # criteria 2: final step feedback is correct
        try:
            final_step = step_feedback_pairs[-1][0]
        except IndexError as e:
            print(e)
            print(step_feedback_pairs)
            print(feedback)
            continue
        all_possible_guess = extract_all_func(final_step)
        is_wrong_step_might_contain_correct_answer = False
        if all_possible_guess is None:
            print("all_possible_guess is None")
            print("final_step:", final_step)
            continue
        for n in all_possible_guess:
            if is_same(n, ref):
                is_wrong_step_might_contain_correct_answer = True
                break
        if is_wrong_step_might_contain_correct_answer:
            print("wrong_step_might_contain_correct_answer")
            pprint(feedback)
            print("wrong step:", item['wrong_code'])
            print("step_feedback_pairs:", step_feedback_pairs)
            print("final_step:", final_step)
            print("all_possible_guess:", all_possible_guess)
            print("ref:", ref)
            print("==="*10)
            wrong_step_might_contain_correct_answer += 1
            continue

        final_step_feedback = step_feedback_pairs[-1][1]
        all_numbers = extract_all_func(final_step_feedback)
        is_final_step_feedback_correctness_failed = True
        if all_numbers is None:
            print("all_numbers is None")
            print("final_step_feedback:", final_step_feedback)
            continue
        for n in all_numbers:
            if is_same(n, ref):
                is_final_step_feedback_correctness_failed = False
                break
        if is_final_step_feedback_correctness_failed:
            print("final_step_feedback_correctness_failed!")
            print("final_step_feedback:", final_step_feedback)
            print("all_numbers", all_numbers)
            print("ref", ref)
            print("==="*10)
            final_step_feedback_correctness_failed += 1
            continue

        # post processing: replace answer 1/2, remove final step feedback
        step_feedback_pair_str = '\n\n'.join([f"{gen_step}\n{gen_feedback}" for gen_step, gen_feedback in step_feedback_pairs[:-1]])
        step_feedback_pair_str += f"\n\n{final_step}"
        prefilterted_step_feedbacks.append(step_feedback_pair_str)
        after_prefilter_step_feedback_pairs_cnt += 1
    item["prefilterted_step_feedbacks"] = prefilterted_step_feedbacks

if use_cot_ft:
    outfile_name = f"logs/{TASK}/prefiltered_feedbacks_cot_ft_diff_{deployment_id}.jsonl"
else:
    outfile_name = f"logs/{TASK}/prefiltered_feedbacks_diff_{deployment_id}.jsonl"
print(outfile_name)
with open(outfile_name, 'w') as writer:
    for item in items:
        writer.write(json.dumps(item)+'\n')

print("total items", len(items))
total_feedbacks = len(items) * len(items[0]['feedbacks'])
print("total feedbacks", total_feedbacks)

print("empty_step_feedback_pairs", empty_step_feedback_pairs, empty_step_feedback_pairs / total_feedbacks)
print("mismatched_step_feedback", mismatched_step_feedback, mismatched_step_feedback / total_feedbacks)
print("copy_failed", copy_failed, copy_failed / total_feedbacks)
print("wrong_step_might_contain_correct_answer", wrong_step_might_contain_correct_answer, wrong_step_might_contain_correct_answer / total_feedbacks)
print("final_step_feedback_correctness_failed", final_step_feedback_correctness_failed, final_step_feedback_correctness_failed / total_feedbacks)
print("after_prefilter_step_feedback_pairs_cnt", after_prefilter_step_feedback_pairs_cnt, after_prefilter_step_feedback_pairs_cnt / total_feedbacks)
