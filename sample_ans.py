from tqdm import tqdm
import re
import os
import json
import random
import datetime
from copy import deepcopy
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
parser.add_argument("--model", type=str)
parser.add_argument("--split", type=str, default='train')
parser.add_argument("--limit", type=int, default=-1)
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--version", type=int, default=0)
parser.add_argument("--num_generations", type=int, default=5)
parser.add_argument("--greedy", action='store_true')
args = parser.parse_args()

IS_DEBUG = False
TASK = args.task
split = args.split
total_test_examples = args.limit
start_data_idx = args.start_idx
use_greedy_decoding = args.greedy
num_generations = args.num_generations
version = args.version

deployment_id = args.model

CHAT = True
# CHAT = "chat" in deployment_id or "Instruct" in deployment_id or "gpt" in deployment_id
EOS = TASK2EOS[TASK]

llama_model, llama_tokenizer = get_model_tokenizer(deployment_id)
os.makedirs(f"logs/{TASK}/", exist_ok=True)
if split == "test":
    save_dir = f"logs/{TASK}/generations_{deployment_id.replace('/', '-')}_{start_data_idx}_{total_test_examples}_test.jsonl"
else:
    save_dir = f"logs/{TASK}/generations_{deployment_id.replace('/', '-')}_{start_data_idx}_{total_test_examples}.jsonl"
if use_greedy_decoding:
    assert num_generations == 1
    save_dir += "_greedy"
if version > 0:
    save_dir += f".version{version}"

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
def openai_completion_wrapper(contexts, is_debug=IS_DEBUG):
    """OpenAI API wrapper, if network error then retry 3 times"""
    if is_debug:
        return random.choice(["disagree", "needs clarification"])
    else:
        if "gpt" in deployment_id:
            # for ctx in contexts:
            #     print(ctx)
            # print('='*50)
            completions = openai.ChatCompletion.create(
                              engine=deployment_id,
                              messages=contexts,
                              temperature=TEMP,
                              n=num_generations)
            usage_statistics.append(completion)
            return completions["choices"][0]["message"]["content"].replace(EOS, "")
        else:
            completions = llama_gen(contexts, llama_model, llama_tokenizer, chat=CHAT, temperature=TEMP, num_return_sequences=num_generations, EOS=EOS)
            # completions = [c.replace("### END", "").replace("# END", "") for c in completions]
            return completions
# 
# def read_jsonl(path: str):
#     with open(path) as fh:
#         return [json.loads(line) for line in fh.readlines() if line]
# 
# def read_demos(file_name):
#     with open(file_name, 'r') as reader:
#         text = reader.read()
#     # if CHAT:
#     #     text = text.replace("### END ###\n", "")
#     utters = text.split("===\n")
#     ACTOR_USER_INITIAL_DEMOS = []
#     for i in range(0, len(utters), 2):
#         ACTOR_USER_INITIAL_DEMOS.append({"role": "user", "content": utters[i]})
#         ACTOR_USER_INITIAL_DEMOS.append({"role": "assistant", "content": utters[i+1]})
#     return ACTOR_USER_INITIAL_DEMOS
# 

if __name__ == "__main__":

    if TASK in ["gsm8k", "gsm8k_nl", "stqa", "csqa", "ld", "math", "riddlesense", "qasc"]:
        # questions = [q for q in load_dataset("gsm8k", "main")[split]]
        with open(f"data/{TASK}/{split}_shuf.jsonl", "r") as reader:
            questions = [json.loads(l) for l in reader]
        INPUT_FIELD = "question"
        OUTPUT_FIELD = "answer"

        ACTOR_USER_INITIAL_DEMOS = read_demos(f"data/{TASK2TRAIN[TASK]}/init.txt")
        ACTOR_USER_INITIAL_PROMPT = TASK2ACTOR_USER_INITIAL_PROMPT[TASK]

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
    if len(questions) == 0:
        print("No remaining questions. Done!")
        exit(0)
    with open(save_dir, 'a', buffering=1) as writer: # line buffered
        for data_idx, data in enumerate(tqdm(questions)):
            question = data[INPUT_FIELD]
            answer = data[OUTPUT_FIELD]

            actor_contexts = []
            actor_contexts += ACTOR_USER_INITIAL_DEMOS
            actor_contexts += [{"role": "user", "content": ACTOR_USER_INITIAL_PROMPT.replace("{question}", question)}]
            # pprint(actor_contexts)

            cots = openai_completion_wrapper(actor_contexts)
            data["generated_answers"] = cots
            writer.write(json.dumps(data)+'\n')
    print("Done!")
