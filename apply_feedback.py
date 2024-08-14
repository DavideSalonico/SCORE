from tqdm import tqdm
import re
import os
import json
import random
import datetime
from copy import deepcopy
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

###################TODO#########################
################################################
#    replace Answer 1 with previous version    #
#    replace Answer 2 with revised version     #
################################################

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--method", type=str)
parser.add_argument("--limit", type=int, default=-1)
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--num_generations", type=int, default=2)
parser.add_argument("--greedy", action='store_true')
parser.add_argument("--cot_ft", action='store_true')
args = parser.parse_args()

IS_DEBUG = False
TASK = args.task
METHOD = args.method
assert METHOD in ['direct', 'diff'] # direct: Q+A->F; diff: Q+A1+A2->F
total_test_examples = args.limit
start_data_idx = args.start_idx
use_greedy_decoding = args.greedy
use_cot_ft = args.cot_ft
num_generations = args.num_generations

deployment_id = args.model

CHAT = True
# CHAT = "chat" in deployment_id or "Instruct" in deployment_id or "gpt" in deployment_id

llama_model, llama_tokenizer = get_model_tokenizer(deployment_id)
if use_cot_ft:
    save_dir = f"logs/{TASK}/corrections_cot_ft_{METHOD}_{deployment_id.replace('/', '-')}_{start_data_idx}_{total_test_examples}.jsonl"
else:
    save_dir = f"logs/{TASK}/corrections_{METHOD}_{deployment_id.replace('/', '-')}_{start_data_idx}_{total_test_examples}.jsonl"
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
            return completions["choices"][0]["message"]["content"].replace("### END ###", "")
        else:
            completions = llama_gen(contexts, llama_model, llama_tokenizer, chat=CHAT, temperature=TEMP, num_return_sequences=num_generations)
            # completions = [c.replace("### END", "").replace("# END", "") for c in completions]
            return completions

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def read_demos(file_name):
    with open(file_name, 'r') as reader:
        text = reader.read()
    # if CHAT:
    #     text = text.replace("### END ###\n", "")
    utters = text.split("===\n")
    CRITIC_DEMOS = []
    for i in range(0, len(utters), 2):
        CRITIC_DEMOS.append({"role": "user", "content": utters[i]})
        CRITIC_DEMOS.append({"role": "assistant", "content": utters[i+1]})
    return CRITIC_DEMOS


if __name__ == "__main__":

    if TASK in ["gsm8k", "gsm8k_nl", "csqa"]:
        # with open(f"logs/{TASK}/feedbacks_{METHOD}_{deployment_id.replace('/', '-')}_{start_data_idx}_{total_test_examples}.jsonl", 'r') as reader:
        if use_cot_ft:
            infile = f"logs/{TASK}/prefiltered_feedbacks_cot_ft_{METHOD}_{deployment_id.replace('/', '-')}.jsonl"
        else:
            infile = f"logs/{TASK}/prefiltered_feedbacks_{METHOD}_{deployment_id.replace('/', '-')}.jsonl"
        with open(infile, 'r') as reader:
            questions = [json.loads(l) for l in reader]
        INPUT_FIELD = "question"
        OUTPUT_FIELD = "answer"
        FEEDBACK_FIELD = "feedbacks" if TASK == "gsm8k" else "prefilterted_step_feedbacks"
        if total_test_examples != -1:
            questions = questions[:total_test_examples]
        questions = questions[start_data_idx:]


        CRITIC_DEMOS = read_demos(f"data/{TASK}/iterate_answer.txt")
        CRITIC_PROMPT = TASK2CRITIC_PROMPT[TASK] 
    else:
        raise ValueError("unsupported task!")


    print(save_dir)
    with open(save_dir, 'w') as writer:
        for data_idx, data in enumerate(tqdm(questions)):
            question = data[INPUT_FIELD]
            answer = data[OUTPUT_FIELD]
            wrong_code = data['wrong_code']
            for feedback in data[FEEDBACK_FIELD]:
                if TASK == "gsm8k":
                    feedback = '\n'.join([l.replace("Answer 1", "the provided code").replace("Answer 2", "the correct code") for l in feedback.split('\n') if l.strip() != "" and l.startswith("• ")])

                critic_contexts = []
                critic_contexts += CRITIC_DEMOS
                critic_contexts += [{"role": "user", "content": CRITIC_PROMPT.replace("{question}", question).replace("{wrong_code}", wrong_code).replace("{feedback}", feedback)}]
                # pprint(critic_contexts)
                cots = openai_completion_wrapper(critic_contexts)
                json_item = {"question":question, "answer":answer, "reference":data['reference'], "wrong_code":wrong_code, "corrections":cots, "feedback":feedback}
                # pprint(json_item)
                writer.write(json.dumps(json_item)+'\n')
    print("Done!")
