from tqdm import tqdm
import re
import os
import json
import random
import datetime
from copy import deepcopy
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
import argparse
from datasets import load_dataset

from pprint import pprint
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
import logging

from llama_inference import llama_gen 
from util import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str)
parser.add_argument("--model", type=str, help='used for generating input solutions to be corrected, e.g. llama-13b')
parser.add_argument("--verifier", type=str, help='used for generating verification, e.g. gpt-4')
parser.add_argument("--split", type=str, default='train')
parser.add_argument("--limit", type=int, default=-1)
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--num_generations", type=int, default=1)
parser.add_argument("--greedy", action='store_true')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--self_refine", action='store_true')
parser.add_argument("--cot_ft_as_input", action='store_true')
args = parser.parse_args()

IS_DEBUG = args.debug
TASK = args.task
split = args.split
total_test_examples = args.limit
start_data_idx = args.start_idx
use_greedy_decoding = args.greedy
num_generations = args.num_generations
self_refine = args.self_refine
cot_ft = args.cot_ft_as_input

"""
if not self_refine:
    assert cot_ft, "sample verification should use rft model's solution as input! add args `--cot_ft`"
else:
    assert not cot_ft, "self-refine prompting should not use rft model's solution as input! remove args `--cot_ft`"
"""

deployment_id = args.verifier
base_model = args.model.replace('/', '-')

openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_base = os.environ.get("OPENAI_ENDPOINT")

openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future

CHAT = True
# CHAT = "chat" in deployment_id or "Instruct" in deployment_id or "gpt" in deployment_id
EOS = TASK2EOS[TASK]

llama_model, llama_tokenizer = get_model_tokenizer(deployment_id)
        
if self_refine:
    if cot_ft:
        base_dir = f"predictions/{TASK}_{split}_cot_ft_as_input_feedback4_refinement_1samples_self_refine_{base_model}"
    else:
        base_dir = f"predictions/{TASK}_{split}_feedback4_refinement_1samples_self_refine_{base_model}"
else:
    if cot_ft:
        base_dir = f"checkpoint/verifier/{TASK}/cot_ft_train_rationale_{base_model}_verified_by_{deployment_id.replace('/', '-')}/"
    else:
        base_dir = f"checkpoint/verifier/{TASK}/{base_model}_verified_by_{deployment_id.replace('/', '-')}/"
if split == "test":
    save_dir = os.path.join(base_dir, "generated_predictions_test.jsonl")
else:
    save_dir = os.path.join(base_dir, "generated_predictions.jsonl")
os.makedirs(base_dir, exist_ok=True)

if use_greedy_decoding:
    assert num_generations == 1
    save_dir += "_greedy"

TEMP = 0.0 if use_greedy_decoding else 0.7
"""
Below are functions calling different APIs. All stops after three retries. 
For the first two retries, if there is a network error, wait for 3 seconds.
For the third retry, if there is a network error, wait for 5 seconds.
"""
STOP_AFTER_ATTEMPT=4
usage_statistics = []

@retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT), 
        wait=wait_chain(*[wait_fixed(3) for i in range(2)] +
                       [wait_fixed(5) for i in range(1)]))
def openai_completion_wrapper(contexts):
    """OpenAI API wrapper, if network error then retry 3 times"""
    if "gpt" in deployment_id:
        # for ctx in contexts:
        #     print(ctx)
        # print('='*50)
        try:
            completions = openai.ChatCompletion.create(
                              engine=deployment_id,
                              messages=contexts,
                              temperature=TEMP,
                              n=num_generations)
            usage_statistics.append(completions['usage'])
        except Exception as e:
            print(e)
            contents = ["ERROR" for i in range(num_generations)]
            return contents

        contents = []
        for c in completions["choices"]:
            try:
                contents.append(c["message"]["content"])
                # return [c["message"]["content"] for c in completions["choices"]]
            except KeyError:
                print("KeyError in message chat completions.")
                print(json.dumps(c, indent=4))
                contents.append("ERROR")
                # return ["ERROR" for i in range(num_generations)]
        return contents
    else:
        completions = llama_gen(contexts, llama_model, llama_tokenizer, chat=CHAT, temperature=TEMP, num_return_sequences=num_generations, EOS=EOS)
        # completions = [c.replace("### END", "").replace("# END", "") for c in completions]
        return completions


if __name__ == "__main__":

    if TASK in ["gsm8k", "gsm8k_nl", "stqa", "csqa", "qasc", "riddlesense", "math"]:
        # questions = [q for q in load_dataset("gsm8k", "main")[split]]
        if cot_ft:
            infile = f"data/{TASK}/verifier/{base_model}/{split}_cot_ft.json"
        else:
            infile = f"data/{TASK}/verifier/{base_model}/{split}.json"

        with open(infile, 'r') as reader:
            questions = [json.loads(l) for l in reader]

        INPUT_FIELD = "question"
        OUTPUT_FIELD = "answer"

        if self_refine:
            demo_file = f"data/{TASK2TRAIN[TASK]}/self_refine.txt"
            VERIFY_PROMPT = TASK2REFINER_PROMPT[TASK]
        else:
            demo_file = f"data/{TASK2TRAIN[TASK]}/verify.txt"
            VERIFY_PROMPT = TASK2VERIFY_PROMPT[TASK]
        VERIFY_DEMOS = read_demos(demo_file)

    else:
        raise ValueError("unsupported task!")


    if total_test_examples != -1:
        questions = questions[:total_test_examples]

    questions = questions[start_data_idx:]
    print(save_dir)
    if os.path.exists(save_dir):
        # resume generations
        with open(save_dir, 'r') as reader:
            completed_items_cnt = len([l for l in reader])
            questions = questions[completed_items_cnt:]
    feedback_instruction = TASK_FEEDBACK_TYPE2FEEDBACK_INSTR[TASK][4]
    with open(save_dir, 'a', buffering=1) as writer: # line buffered
        for data_idx, data in enumerate(tqdm(questions)):
            ctxs = []
            ctxs += VERIFY_DEMOS
            if self_refine:
                content = VERIFY_PROMPT.replace("{text}", data['text'].replace("Is this solution correct or not?", "").strip()).replace("{feedback_instruction}", feedback_instruction)
            else:
                content = VERIFY_PROMPT.replace("{text}", data['text'])
            ctxs += [{"role": "user", "content": content}]
            if IS_DEBUG:
                pprint(ctxs)

            verifications = openai_completion_wrapper(ctxs)
            if self_refine:
                data["predict"] = verifications[0] if use_greedy_decoding else verifications
            else:
                data["generated_verifications"] = verifications
                data["label"] = data["target"]
                del data['target']
                preds = []
                for v in verifications:
                    if "**incorrect**" in v:
                        preds.append(0)
                    elif "**correct**" in v:
                        preds.append(1)
                    elif "ERROR" == v:
                        preds.append(-2)
                    else:
                        print("no **incorrect** or **correct** in v:")
                        print(v)
                        if "incorrect" in v:
                            preds.append(0)
                        else:
                            preds.append(-1)
                    """
                    if "YES" in v and "NO" in v:
                        print("YES and NO in v:")
                        print(v)
                        preds.append(-1)
                    elif "YES" in v:
                        preds.append(1)
                    elif "NO" in v:
                        preds.append(0)
                    else:
                        print("neither YES nor NO in v:")
                        print(v)
                        preds.append(-2)
                    """

                data['extracted_predictions'] = preds
                filtered_preds = [p for p in preds if p == 0 or p == 1]
                if len(filtered_preds) == 0:
                    data['predict'], data['confidence'] = 1, 1.0
                else:
                    data['predict'], data['confidence'], _ = majority_voting(filtered_preds, data["label"])
                    data['predict'] = int(data['predict'])
            writer.write(json.dumps(data)+'\n')
    
    if "gpt" in deployment_id:
        summary_usage(usage_statistics, deployment_id)
    print("Done!")
