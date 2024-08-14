from tqdm import tqdm
import openai
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
parser.add_argument("--method", type=str)
parser.add_argument("--limit", type=int, default=-1)
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--num_generations", type=int, default=2)
parser.add_argument("--greedy", action='store_true')
parser.add_argument("--cot_ft", action='store_true') # use sampled solutions from cot-finetuned (RFT) model instead of the base model
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

openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_base = os.environ.get("OPENAI_ENDPOINT")
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future

llama_model, llama_tokenizer = get_model_tokenizer(deployment_id)
if use_cot_ft:
    save_dir = f"logs/{TASK}/feedbacks_more_cot_ft_{METHOD}_{deployment_id.replace('/', '-')}_{start_data_idx}_{total_test_examples}.jsonl"
else:
    save_dir = f"logs/{TASK}/feedbacks_{METHOD}_{deployment_id.replace('/', '-')}_{start_data_idx}_{total_test_examples}.jsonl"
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
#     CRITIC_DEMOS = []
#     for i in range(0, len(utters), 2):
#         CRITIC_DEMOS.append({"role": "user", "content": utters[i]})
#         CRITIC_DEMOS.append({"role": "assistant", "content": utters[i+1]})
#     return CRITIC_DEMOS
# 

if __name__ == "__main__":

    if TASK in ["gsm8k", "gsm8k_nl", "csqa", "ld"]:
        if use_cot_ft:
            infile = f"logs/{TASK}/more_cot_ft_generations_{deployment_id.replace('/', '-')}_score.jsonl"
        else:
            infile = f"logs/{TASK}/generations_{deployment_id.replace('/', '-')}_score.jsonl"

        with open(infile, 'r') as reader:
            questions = [json.loads(l) for l in reader]

        if use_cot_ft:
            infile = infile.replace("more_cot_ft", "cot_ft")
            if os.path.exists(infile):
                # save_dir = save_dir.replace("cot_ft", "more_cot_ft")
                with open(infile, 'r') as reader:
                    more_questions = [json.loads(l) for l in reader]
                    assert len(questions) == len(more_questions)
                for question, more_question in zip(questions, more_questions):
                    question['all_score'] = question['score'] + more_question['score']
                    question['generated_answers'].extend(more_question['generated_answers'])
                    
        INPUT_FIELD = "question"
        OUTPUT_FIELD = "answer"
        
        if METHOD == "direct":
            CRITIC_DEMOS = read_demos(f"data/{TASK}/feedback.txt")
            if TASK == "gsm8k":
                CRITIC_PROMPT = "# Q: {question}\n# Please answer this question by implementing a `solution()` Python function that returns the result.\n\n{wrong_code}\n\n# There are errors in the code above because of lack of understanding of the question. Please provide feedback that helps correct the errors in the code above. Specifically, point out the incorrect line of code, explain why, and provide specific suggestions for error correction.\n"
            elif TASK == "gsm8k_nl":
                CRITIC_PROMPT = "Q: {question}\nExplain your reasoning step by step. Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response.\n\n{wrong_code}\n\nIf there are any issues with the answer provided, please identify these errors in the reasoning steps.\n"
        elif METHOD == "diff":
            CRITIC_DEMOS = read_demos(f"data/{TASK}/diff_feedback.txt")
            if TASK == "gsm8k":
                CRITIC_PROMPT = "# Q: {question}\n# Please answer this question by implementing a `solution()` Python function that returns the result.\n\n# [Answer 1] (Incorrect):\n{wrong_code}\n\n# [Answer 2] (Correct):\n{right_code}\n\n# There are errors in Answer 1 because of lack of understanding of the question. Please use Answer 2 as a reference for the correct approach, and provide feedback that helps correct the errors in Answer 1. Specifically, point out the incorrect line of code, explain why, and use Answer 2 to provide specific suggestions for error correction.\n"
            elif TASK in ["gsm8k_nl", "csqa", "ld"]:
                CRITIC_PROMPT = "Q: {question}\n\nAnswer 1 (Incorrect):\n{wrong_code}\n\nAnswer 2 (Correct):\n{right_code}\n\nThere are reasoning errors in Answer 1. Please go through each step in Answer 1, use Answer 2 as a reference for the correct approach, and provide feedback that helps correct the errors in Answer 1. End your response with [END].\n"
        else:
            raise ValueError("unsupported method!")
    else:
        raise ValueError("unsupported task!")


    if total_test_examples != -1:
        questions = questions[:total_test_examples]

    questions = questions[start_data_idx:]
    print(save_dir)

    with open(save_dir, 'a', buffering=1) as writer:
        for data_idx, data in enumerate(tqdm(questions)):
            question = data[INPUT_FIELD]
            answer = data[OUTPUT_FIELD]
            generated_answers = data['generated_answers']
            if "all_score" in data:
                score = data['all_score']
            else:
                score = data['score']

            # sampling value-improving pairs
            value_improving_idx_pairs = []
            if METHOD == "diff":
                for i in range(len(score)):
                    for j in range(len(score)):
                        if 'all_score' in data and i < len(score) / 2 and j < len(score) / 2:
                            continue
                        if score[i] == 0 and data['extracted_answers'][i] is not None and score[j] == 1:
                            value_improving_idx_pairs.append((i, j))
                """
                if TASK in ["gsm8k", "gsm8k_nl", "csqa"]:
                    if "all_score" in data:
                        assert len(value_improving_idx_pairs) == len([s for s in data['all_score'] if s == 0]) * len([s for s in data['all_score'] if s == 1]) - len([s for s in data['score'] if s == 0]) * len([s for s in data['score'] if s == 1]) 
                    else:
                        assert len(value_improving_idx_pairs) == len([s for s in score if s == 0]) * len([s for s in score if s == 1])
                """
            elif METHOD == "direct":
                if 1 in score:
                    value_improving_idx_pairs = [(idx, 0) for idx, s in enumerate(score) if s == 0]
                else:
                    value_improving_idx_pairs = []

            # pprint(value_improving_idx_pairs)
            for wrong_idx, right_idx in value_improving_idx_pairs:
                wrong_code = clean_cot(generated_answers[wrong_idx])
                right_code = clean_cot(generated_answers[right_idx])
                critic_contexts = []
                critic_contexts += CRITIC_DEMOS
                critic_contexts += [{"role": "user", "content": CRITIC_PROMPT.replace("{question}", question).replace("{wrong_code}", wrong_code).replace("{right_code}", right_code)}]
                # pprint(critic_contexts)
                cots = openai_completion_wrapper(critic_contexts)
                json_item = {"question":question, "answer":answer, "reference":data['reference'], "wrong_code":wrong_code, "right_code":right_code, "feedbacks":cots}
                # pprint(json_item)
                writer.write(json.dumps(json_item)+'\n')
    print("Done!")
