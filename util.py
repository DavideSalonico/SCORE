import re
import os
import json
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

hf_token = "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"
TASK_LIST = ["gsm8k", "gsm8k_nl", "csqa", "ld", "svamp", "math", "riddlesense", "qasc"]
MODEL_LIST = ["meta-llama/Llama-2-13b-chat-hf", "google/gemma-7b-it", "meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2"]

OPENAI_API_PRICES = {
        "gpt-35-turbo" : {
            "prompt_tokens" : 0.0015 / 1000,
            "completion_tokens" : 0.002 / 1000,
            },
        "gpt-4" : {
            "prompt_tokens" : 0.03 / 1000,
            "completion_tokens" : 0.06 / 1000,
            },
        }
TASK2RANGE = { # train + dev (seen data)
        "gsm8k" : range(0, 7500, 500),
        "gsm8k_nl" : range(0, 7500, 500),
        "csqa" : range(0, 10000, 500),
        "ld" : range(0, 800, 50),
        "math" : range(0, 50, 50),
        "riddlesense" : range(0, 50, 50),
        "qasc" : range(0, 50, 50),
        }
TASK2TEST_RANGE = {
        "gsm8k" : range(0, 1500, 500),
        "gsm8k_nl" : range(0, 1500, 500),
        "csqa" : range(0, 1500, 500),
        "ld" : range(0, 700, 50),
        "math" : range(0, 200, 50),
        "riddlesense" : range(0, 1100, 100),
        "qasc" : range(0, 1000, 100),
        }
TASK2INTERVAL = {
        "gsm8k" : 500,
        "gsm8k_nl" : 500,
        "csqa" : 500,
        "ld" : 50,
        "math" : 50,
        "riddlesense" : 100,
        "qasc" : 100,
        }

TASK2DEV_SIZE = {
        "gsm8k" : 6500,
        "gsm8k_nl" : 6500,
        "csqa" : 8500,
        "ld" : 600,
        "math" : 0,
        "riddlesense" : 0,
        "qasc" : 0,
        }


TASK2EOS = {"gsm8k" : "return"}
for task in TASK_LIST:
    TASK2EOS[task] = "[END]"
# "gsm8k_nl" : "[END]", "stqa" : "[END]", "csqa" : "[END]", "ld" : "[END]", ""}
TASK2ACTOR_USER_INITIAL_PROMPT = {
        "gsm8k" : "# Q: {question}\n# Can you answer this question by implementing a `solution()` Python function that returns the result? Note that your answer should only contain the code.\n",
        "gsm8k_nl" : "Q: {question}\nExplain your reasoning step by step. Your final answer should be a single numerical number, in the form \\boxed{answer}. End your response with [END].\n",
        "stqa" : "Q: Yes or no: {question}\nExplain your reasoning step by step. Your final answer should be either \"yes\" or \"no\", in the form **yes** or **no**. End your response with [END].\n",
        "csqa" : "Q: {question}\nExplain your reasoning step by step. Your final answer should be a single letter from A to E, in the form (answer). End your response with [END].\n",
        "ld" : "Q: {question}\nExplain your reasoning step by step. Let \"??\" represent 0 or more objects, and \"?\" represent exactly 1 object. Your final answer should be a single letter in the form (answer). End your response with [END].\n",
        }
TASK2CRITIC_PROMPT = {
        "gsm8k" : "# Q: {question}\n# Please answer this question by implementing a `solution()` Python function that returns the result.\n\n{wrong_code}\n\n# There are errors in the code above because of lack of understanding of the question. What are the errors?\n\n{feedback}\n\n# Can you correct the errors in the code? Please ensure to start and finish your code with ```. Note that your answer should only contain the code and nothing else.\n",
        "gsm8k_nl" : "Q: {question}\n\n{feedback}\n\nCan you correct the errors in your reasoning based on the feedback provided? Your final answer should be a single numerical number, in the form \\boxed{answer}. End your response with [END].\n",
        "csqa" : "Q: {question}\n\n{feedback}\n\nCan you correct the errors in your reasoning based on the feedback provided? Your final answer should be a single letter from A to E, in the form (answer). End your response with [END].\n",
        }
TASK2ALREADY_CORRECT_MSG = {
        "gsm8k" : "`def solution()`: This solution is correct.",
        "gsm8k_nl" : "The final answer is correct.",
        "csqa" : "The final answer is correct.",
        }
DEFAULT_FEEDBACK_INSTR = {
            # original
            0 : "Does the solution contains any errors? If yes, please go through each step in the solution, and provide feedback that helps correct the errors.",
            # 1 (binary classifier)
            1 : "Is the solution correct?",
            # 2 (short feedback)
            2 : "Does the solution contains any errors? If yes, please go through each step in the solution, and provide feedback that helps correct the errors.",
            # 3 (final answer feedback)
            3 : "Does the solution contains any errors? If yes, please go through each step in the solution, and provide feedback that helps correct the errors.",
            # 4 (only include the first feedback) & keep reduce the proportion of already-correct feedback
            4 : "Does the solution contains any errors? If yes, please identify the first error step in the solution, and provide feedback that helps correct the error.",
            # 5 (empty feedback)
            5 : "Does the solution contains any errors?",
            }

TASK_FEEDBACK_TYPE2FEEDBACK_INSTR = {t : DEFAULT_FEEDBACK_INSTR for t in TASK_LIST}

DEFAULT_VERIFY_FT_PROMPT = "Question: {question}\n\nSolution:\n{solution}\n\nIs this solution correct or not?"
TASK2VERIFY_FT_PROMPT = {t : DEFAULT_VERIFY_FT_PROMPT for t in TASK_LIST}

DEFAULT_VERIFY_PROMPT = "{text} Answer **correct** or **incorrect** and explain your reasoning. End your response with [END].\n"
# {text} represents Q+Solution+Is this solution correct or not?
TASK2VERIFY_PROMPT = {t : DEFAULT_VERIFY_PROMPT for t in TASK_LIST}

TASK2REFINER_FT_PROMPT = {
        "gsm8k_nl" : "Question: {question}\n\nSolution:\n{solution}\n\n{feedback_instruction} Then refine your reasoning based on the feedback. Your final answer should be a single numerical number, in the form \\boxed{{answer}}.\n\nFeedback:",
        "csqa" : "Question: {question}\n\nSolution:\n{solution}\n\n{feedback_instruction} Then refine your reasoning based on the feedback. Your final answer should be a single letter from A to E, in the form (answer).\n\nFeedback:",
        }
TASK2REFINER_PROMPT = {
        "gsm8k_nl" : "{text}\n\n{feedback_instruction} Then refine your reasoning based on the feedback. Your final answer should be a single numerical number, in the form \\boxed{{answer}}. End your response with [END].\n",
        "csqa" : "{text}\n\n{feedback_instruction} Then refine your reasoning based on the feedback. Your final answer should be a single letter from A to E, in the form (answer). End your response with [END].\n",
        }


def clean_cot(input_str):
    input_str_cleaned = input_str.replace('print(solution())', '')
    input_str_cleaned = re.sub("(\[END|\[|END)$", "", input_str_cleaned)
    input_str_cleaned = input_str_cleaned.replace("[END]", "")
    input_str_cleaned = input_str_cleaned.replace("**", "") # for gemma
    """
    for token in ["[END", "[", "END"]:
        if input_str_cleaned.endswith(token):
            input_str_cleaned = input_str_cleaned.replace(token, "")
            break
    """
    return input_str_cleaned.strip()

def extract_all_numbers(input_str):
    all_numbers = [re.sub(r"[^0-9.]", "", n) for n in re.findall("[0-9,]*\.?[0-9,]+", re.sub("Step \d+:", "", input_str))]
    all_numbers = [n for n in all_numbers if n != ""]
    return all_numbers

def extract_all_choices(input_str):
    all_choices = re.findall("\(([A-Z])\)", input_str)
    return all_choices


def is_same(result, correct_solution, tol = 1e-3):
    if result is None or correct_solution is None:
        return False
    result = str(result)
    correct_solution = str(correct_solution)
    if result.strip() == correct_solution.strip():
        return True
    try:
        result = float(result.strip())
        correct_solution = float(correct_solution.strip())
        return abs(result - correct_solution) < tol
    except:
        return False

def parse_answer(input_str):
    pattern = r"\{\s*([0-9.,$]*)\s*\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    return solution

def parse_choice(input_str):
    pattern = r"\(([a-zA-Z])\)"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = match_str
        if solution:
            break

    return solution




# adapted from https://github.com/wenhuchen/Program-of-Thoughts/blob/main/tool.py#L91
def floatify_ans(ans):
    if ans is None:
        return None
    else:
        try:
            ans = float(ans)
        except Exception:
            ans = str(ans)
    return ans

def majority_voting(preds, ref):
    result_counter = Counter()
    for ans in preds:
        ans = floatify_ans(ans)
        if ans is not None:
            result_counter.update([ans])

    if len(result_counter) > 0:
        most_common_ans_freq = result_counter.most_common(1)[0]
        prediction, confidence = most_common_ans_freq[0], most_common_ans_freq[1] / len(preds)
        """
        print("preds", preds)
        print("ref", ref)
        print("result_counter", result_counter)
        print("most_common_ans_freq", most_common_ans_freq)
        print("prediction", prediction, "confidence", confidence)
        # exit(0)
        """
        correctness = int(is_same(prediction, ref))
    else:
        prediction, confidence, correctness = None, 0, 0
    return prediction, confidence, correctness	


def save_to_file(items, outfile, chunk):
    if "tmp" in outfile:
        return
    if chunk == 1:
        with open(outfile, 'w') as writer:
            json.dump(items, writer, indent=4)
        print(outfile, f"{len(items)} items")
    else:
        # split into 8 slices
        size = len(items) // chunk + 1
        for c in range(chunk):
            outfile_chunk = outfile.replace(".json", f".part{c:02}.json") # f"../LLaMA-Factory/data/{task}_{split}_rationale_part{c:02}_{model}.json"
            interval = items[c*size:c*size+size]
            with open(outfile_chunk, 'w') as writer:
                json.dump(interval, writer, indent=4)
            print(outfile_chunk, f"{len(interval)} items")

def get_correct_generation_ratio(task, model):
    infile = f"logs/{task}/generations_{model}_score.jsonl"
    with open(infile, 'r') as reader:
        items = [json.loads(l) for l in reader]

    scores = []
    for item in items:
        # score = tuple(sorted(item['score']))
        # score = item['score'][0]
        # scores.append(score)
        scores.extend(item['score'])

    # score_counter = Counter(scores)
    # print(infile)
    # print(score_counter)
    correct_cnt = len([s for s in scores if s == 1])
    return correct_cnt / len(scores)

def remove_copied_steps(step_feedback, add_final_step_feedback=False, return_feedback_for_first_error_step=False):
    feedbacks = re.findall("Feedback: (.*)", step_feedback)
    ret = '\n'.join([f"Step {i+1}: {f}" for i, f in enumerate(feedbacks)])
    if add_final_step_feedback:
        ret += f"\nStep {len(feedbacks)+1}: The final answer is wrong!"
    elif return_feedback_for_first_error_step:
        for i, f in enumerate(feedbacks):
            if not f.endswith(" correct."):
                ret = f"Step {i+1}: {f}"
                break
    return ret

def summary_usage(usage_statistics, deployment_id):
    total_cost = 0
    print(f"total queries: {len(usage_statistics)}")
    for k in ["completion_tokens", "prompt_tokens", "total_tokens"]:
        sum_tokens = sum([u[k] for u in usage_statistics])
        avg_tokens = sum_tokens / len(usage_statistics)
        total_cost += OPENAI_API_PRICES[deployment_id].get(k, 0) * sum_tokens
        print(f"avg {k}: {avg_tokens:.2f}")
    print(f"total cost: {total_cost} dollars")

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def read_demos(file_name):
    with open(file_name, 'r') as reader:
        text = reader.read()
    # if CHAT:
    #     text = text.replace("### END ###\n", "")
    utters = text.split("===\n")
    ctxs = []
    for i in range(0, len(utters), 2):
        ctxs.append({"role": "user", "content": utters[i]})
        ctxs.append({"role": "assistant", "content": utters[i+1]})
    return ctxs

def get_model_tokenizer(deployment_id):
    llama_model, llama_tokenizer = None, None
    if "llama" in deployment_id.lower() or "gemma" in deployment_id.lower() or "mistral" in deployment_id.lower():
        model_id = deployment_id
        llama_tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        if "34b" in model_id or "70b" in model_id:
            quantization_config = BitsAndBytesConfig(
               # 4bit quantization config
               # load_in_4bit=True,
               # bnb_4bit_quant_type="nf4",
               # bnb_4bit_use_double_quant=True,
               # bnb_4bit_compute_dtype=torch.float16
               # 8bit quantization config
               load_in_8bit=True,
            )
            llama_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                token=hf_token
            )
        else:
            llama_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                # quantization_config=quantization_config,
                device_map="auto",
                torch_dtype="auto", # torch.float16 if "llama" in model_id.lower() else torch.bfloat16,
                token=hf_token
            )
    return llama_model, llama_tokenizer


TASK2PARSING_FUNC = {
        "gsm8k" : parse_answer,
        "gsm8k_nl" : parse_answer,
        "csqa" : parse_choice,
        "ld" : parse_choice,
        }
TASK2EXTRACTING_ALL_FUNC = {
        "gsm8k" : extract_all_numbers,
        "gsm8k_nl" : extract_all_numbers,
        "csqa" : extract_all_choices,
        "ld" : parse_choice,
        }
TASK2TRAIN = {
        "gsm8k" : "gsm8k",
        "gsm8k_nl" : "gsm8k_nl",
        "csqa" : "csqa",
        "ld" : "ld",
        "svamp" : "gsm8k_nl",
        "math" : "gsm8k_nl",
        "riddlesense" : "csqa",
        "qasc" : "csqa",
        }

for TASK2STH in [TASK2ACTOR_USER_INITIAL_PROMPT, TASK2CRITIC_PROMPT, TASK2ALREADY_CORRECT_MSG, TASK2REFINER_FT_PROMPT, TASK2REFINER_PROMPT, TASK2PARSING_FUNC, TASK2EXTRACTING_ALL_FUNC]:
    for TASK in TASK2TRAIN:
        TRAIN = TASK2TRAIN[TASK]
        if TRAIN in TASK2STH:
            TASK2STH[TASK] = TASK2STH[TRAIN]
