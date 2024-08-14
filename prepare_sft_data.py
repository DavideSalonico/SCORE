import json
import os
import argparse
import random

from util import *

valid_stages = [
        "sft", 
        "sft_balanced", 
        "sft_rebalanced", 
        "predict_feedback", 
        "predict_feedback_cot_ft_as_input",
        "predict_feedback_ft_rationale", 
        "predict_refinement",
        "predict_cot_ft_as_input_refinement",
        "predict_balanced_refinement",
        "predict_rebalanced_refinement",
        "predict_cot_ft_as_input_rebalanced_refinement",
        ]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str, choices=TASK_LIST)
parser.add_argument("--model", type=str, choices=MODEL_LIST)
parser.add_argument("--stage", type=str, choices=valid_stages)
parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
parser.add_argument("--chunk", type=int, default=1)
parser.add_argument("--sample", type=int, default=1)
parser.add_argument("--feedback_type", type=int, default=4)
parser.add_argument("--ratio_correct", type=float, default=-1)
parser.add_argument("--cot_ft", action='store_true', help='used with sft stage')
args = parser.parse_args()

task = args.task
model = args.model.replace("/", '-')
stage = args.stage 
split = args.split
chunk = args.chunk
SAMPLE_SIZE = args.sample
feedback_type = args.feedback_type
correct_generation_ratio = args.ratio_correct
use_cot_ft = args.cot_ft
if "rebalanced" in stage and args.ratio_correct == -1:
    correct_generation_ratio = get_correct_generation_ratio(task, model)

already_correct_message = TASK2ALREADY_CORRECT_MSG[task]


print("stage:", stage)

def gen_all_step_correct_feedback(steps):
    split_steps = ["Step" + line for line in re.split("Step", clean_cot(steps)) if line.strip() != "" and line.strip()[0].isdigit()]
    if len(split_steps) == 0:
        # print(steps)
        # print("==="*10)
        return ""
    all_step_correct_feedback = ""
    for s in split_steps[:-1]:
        all_step_correct_feedback += f"{s.strip()}\nFeedback: This step is correct.\n\n"
    final_step = split_steps[-1]
    all_step_correct_feedback += final_step.strip()
    return all_step_correct_feedback.strip()

def is_all_step_correct_feedback(step_feedbacks):
    for step_feedback in re.split("Step", step_feedbacks):
        if "Feedback: This step is correct." not in step_feedback:
            return False
    return True


wrong_code2score = dict()

if stage in ["sft", "sft_balanced", "sft_rebalanced"]:
    assert chunk == 1, f"{stage} stage is generating training data, not inference data. no need to split data into chunks!"
    if stage == "sft":
        outfile_feedback = f"../LLaMA-Factory/data/{task}_filtered_feedback{feedback_type}_diff_{model}.json"
        outfile_refinement = f"../LLaMA-Factory/data/{task}_filtered_refinement{feedback_type}_diff_{model}.json"
        outfile_feedback_refinement = f"../LLaMA-Factory/data/{task}_filtered_feedback{feedback_type}_refinement_diff_{model}.json"
    elif stage == "sft_balanced":
        outfile_feedback = f"../LLaMA-Factory/data/{task}_filtered_balanced_feedback_diff_{model}.json"
        outfile_refinement = f"../LLaMA-Factory/data/{task}_filtered_refinement_diff_{model}.json"
        outfile_feedback_refinement = f"../LLaMA-Factory/data/{task}_filtered_balanced_feedback_refinement_diff_{model}.json"
    elif stage == "sft_rebalanced":
        outfile_feedback = f"../LLaMA-Factory/data/{task}_filtered_rebalanced_feedback{feedback_type}_{correct_generation_ratio}_diff_{model}.json"
        outfile_refinement = f"../LLaMA-Factory/data/{task}_filtered_refinement_diff_{model}.json"
        outfile_feedback_refinement = f"../LLaMA-Factory/data/{task}_filtered_rebalanced_feedback_refinement_diff_{model}.json"
    """
    with open(f"logs/{task}/generations_{model}_score.jsonl", 'r') as reader:
        for l in reader:
            item = json.loads(l)
            for gen_answer, s in zip(item['generated_answers'], item['score']):
                gen_answer = clean_cot(gen_answer) # .replace('print(solution())', '').strip()
                wrong_code2score[gen_answer] = s
    """


    if use_cot_ft:
        infile = f"logs/{task}/corrections_cot_ft_diff_{model}_scores.jsonl"
        outfile_feedback = outfile_feedback.replace("diff", "cot_ft_diff")
        outfile_refinement = outfile_refinement.replace("diff", "cot_ft_diff")
        outfile_feedback_refinement = outfile_feedback_refinement.replace("diff", "cot_ft_diff")
    else:
        infile = f"logs/{task}/corrections_diff_{model}_scores.jsonl"
        
    print("loading from ", infile)
    with open(infile, 'r') as reader:
        items = [json.loads(l) for l in reader]
elif stage in ["predict_feedback", "predict_feedback_cot_ft_as_input", "predict_feedback_ft_rationale", "predict_refinement", "predict_cot_ft_as_input_refinement", "predict_rebalanced_refinement", "predict_cot_ft_as_input_rebalanced_refinement"]:
    outfile_refinement = "data/tmp.json"
    outfile_feedback = "data/tmp.json"
    outfile_feedback_refinement = "data/tmp.json"
    if stage == "predict_feedback_ft_rationale":
        outfile_feedback = f"../LLaMA-Factory/data/{task}_test_feedback_ft_rationale_{model}.json"
        ft_rationale_generations_file = f"../LLaMA-Factory/predictions/gsm8k_ft_rationale_meta-llama-Llama-2-13b-chat-hf/generated_predictions.jsonl"
        with open(ft_rationale_generations_file, 'r') as reader:
            ft_rationale_generations = [json.loads(l) for l in reader]
    elif stage == "predict_feedback":
        outfile_feedback = f"../LLaMA-Factory/data/{task}_{split}_feedback{feedback_type}_{model}.json"
        outfile_feedback_refinement = f"../LLaMA-Factory/data/{task}_{split}_feedback{feedback_type}_refinement_{SAMPLE_SIZE}samples_{model}.json"
    elif stage == "predict_feedback_cot_ft_as_input":
        outfile_feedback = f"../LLaMA-Factory/data/{task}_{split}_cot_ft_as_input_feedback{feedback_type}_{model}.json"
        outfile_feedback_refinement = f"../LLaMA-Factory/data/{task}_{split}_cot_ft_as_input_feedback{feedback_type}_refinement_{SAMPLE_SIZE}samples_{model}.json"
    elif stage == "predict_refinement":
        gen_feedback_file = f"../LLaMA-Factory/predictions/{task}_{split}_feedback{feedback_type}_filtered_feedback{feedback_type}_diff_{model}/generated_predictions.jsonl.00"
        outfile_refinement = f"../LLaMA-Factory/data/{task}_{split}_feedback{feedback_type}_filtered_feedback{feedback_type}_diff_refinement_{model}.json"
    elif stage == "predict_cot_ft_as_input_refinement":
        gen_feedback_file = f"../LLaMA-Factory/predictions/{task}_{split}_cot_ft_as_input_feedback{feedback_type}_filtered_feedback{feedback_type}_diff_{model}/generated_predictions.jsonl.00"
        outfile_refinement = f"../LLaMA-Factory/data/{task}_{split}_cot_ft_as_input_feedback{feedback_type}_filtered_feedback{feedback_type}_diff_refinement_{model}.json"
    elif stage == "predict_rebalanced_refinement":
        gen_feedback_file = f"../LLaMA-Factory/predictions/{task}_{split}_feedback{feedback_type}_filtered_rebalanced_feedback{feedback_type}_{correct_generation_ratio:.2f}_diff_{model}/generated_predictions.jsonl.00"
        outfile_refinement = f"../LLaMA-Factory/data/{task}_{split}_feedback{feedback_type}_filtered_rebalanced_feedback{feedback_type}_{correct_generation_ratio:.2f}_diff_refinement_{model}.json"
    elif stage == "predict_cot_ft_as_input_rebalanced_refinement":
        gen_feedback_file = f"../LLaMA-Factory/predictions/{task}_{split}_cot_ft_as_input_feedback{feedback_type}_filtered_rebalanced_feedback{feedback_type}_{correct_generation_ratio:.2f}_diff_{model}/generated_predictions.jsonl.00"
        outfile_refinement = f"../LLaMA-Factory/data/{task}_{split}_cot_ft_as_input_feedback{feedback_type}_filtered_rebalanced_feedback{feedback_type}_{correct_generation_ratio:.2f}_diff_refinement_{model}.json"

    already_correct_items = []

    items = []
    jsonl_items = []

    generations_file_name = f"logs/{task}/generations_{model}_score.jsonl"
    if split == "test":
        generations_file_name = generations_file_name.replace(".jsonl", "_test.jsonl")
    if SAMPLE_SIZE == 1:
        generations_file_name += "_greedy"
    print(f"loading from {generations_file_name}")
    with open(generations_file_name, 'r') as reader:
        jsonl_items = [json.loads(l) for l in reader]
        if split == "dev":
            dev_size = TASK2DEV_SIZE[task]
            if len(jsonl_items) > dev_size:
                jsonl_items = jsonl_items[dev_size:]

    for sample_idx in range(SAMPLE_SIZE):
        rationale_items = []
        gen_feedback_items = []
        if SAMPLE_SIZE == 1:
            # greedy decoding  results
            # with open("../LLaMA-Factory/predictions/gsm8k_nl_{split}_rationale_meta-llama-Llama-2-13b-chat-hf_rank32_lr5e-5_2337/generated_predictions.jsonl", 'r') as reader:
            if stage == "predict_feedback_cot_ft_as_input":
                with open(f"../LLaMA-Factory/predictions/{task}_{split}_rationale_train_rationale_meta-llama-Llama-2-13b-chat-hf/generated_predictions.jsonl.00", 'r') as reader:
                    rationale_items.extend([json.loads(l) for l in reader])
            else:
                rationale_items = jsonl_items
            if "refinement" in stage:
                print(f"loading generated feedback from {gen_feedback_file}")
                with open(gen_feedback_file, 'r') as reader:
                    gen_feedback_items.extend([json.loads(l) for l in reader])
        else:
            if stage == "predict_feedback_cot_ft_as_input":
                for part_idx in range(16):
                    prediction_file = f"../LLaMA-Factory/predictions/{task}_part{part_idx:02}_{split}_rationale_train_rationale_meta-llama-Llama-2-13b-chat-hf/generated_predictions.jsonl.{sample_idx:02}"
                    if not os.path.exists(prediction_file):
                        print(f"{prediction_file} not exists!")
                        continue
                    with open(prediction_file, 'r') as reader:
                        rationale_items.extend([json.loads(l) for l in reader])
                    if "refinement" in stage:
                        pass
            else:
                rationale_items = jsonl_items
        assert len(jsonl_items) == len(rationale_items), f"len(jsonl_items) = {len(jsonl_items)} while len(rationale_items) = {len(rationale_items)}"
        if "refinement" in stage:
            assert len(jsonl_items) == len(gen_feedback_items), f"len(jsonl_items) = {len(jsonl_items)} while len(gen_feedback_items) = {len(gen_feedback_items)}"

        for idx, (jsonl_item, rationale_item) in enumerate(zip(jsonl_items, rationale_items)):
            item = {
                    "question" : jsonl_item['question'], 
                    "answer" : "",
                    "reference" : "",
                    "wrong_code" : clean_cot(rationale_item['predict']) if stage == "predict_feedback_cot_ft_as_input" else clean_cot(jsonl_item['generated_answers'][sample_idx]),
                    "corrections" : [""],
                    "feedback" : gen_feedback_items[idx]['predict'] if "refinement" in stage else "" ,
                    "scores": [1],
                    }
            if "rebalanced_refinement" in stage:
                condition1 = feedback_type == 0 and is_all_step_correct_feedback(item['feedback'])
                condition2 = "This solution is correct." in item['feedback'] or already_correct_message in item['feedback']
                if condition1 or condition2:
                    pseudo_generated_pred = {
                            "label" : "```\n\n```\n\n### END ###",
                            "predict" : f"{already_correct_message}\nCOPY OF INITIAL SOLUTION:\n{item['wrong_code']}", # skip refinement
                            "idx" : idx,
                            }
                    already_correct_items.append(pseudo_generated_pred)
                else:
                    items.append(item)
            else:
                items.append(item)



    """
    infile_dir = f"../fof/logs/{task}/fof_answer_rounds_1_feedback_rounds_3/few_shot/{model}/" # TODO
    outfile_refinement = "data/tmp.json"
    test_example_cnt = 100
    items = []
    for idx in range(test_example_cnt):
        with open(os.path.join(infile_dir, f"actor{idx}.jsonl"), 'r') as reader:
            jsonl_items = [json.loads(l) for l in reader]
        item = {
                "question" : jsonl_items[0]['content'].split('#')[1].replace("Q:", "").strip(),
                "answer" : "",
                "reference" : "",
                "wrong_code" : ft_rationale_generations[idx]['predict'].replace("```", "").replace("### END ###", "").strip() if stage == "predict_feedback_ft_rationale" else jsonl_items[1]['content'].replace("```", "").strip(),
                "corrections" : [""],
                "feedback" : "",
                "scores": [1],
                }
        items.append(item)
    """
elif stage in ["predict_balanced_refinement"]:
    outfile_feedback = "data/tmp.json"
    if stage == "predict_refinement":
        gen_feedback_file = f"../LLaMA-Factory/predictions/{task}_filtered_feedback_diff_{model}/generated_predictions.jsonl"
        outfile_refinement = f"../LLaMA-Factory/data/{task}_test_refinement_{model}.json"
    elif stage == "predict_balanced_refinement":
        gen_feedback_file = f"../LLaMA-Factory/predictions/{task}_filtered_balanced_feedback_diff_{model}/generated_predictions.jsonl"
        outfile_refinement = f"../LLaMA-Factory/data/{task}_test_balanced_refinement_{model}.json"
    items = []
    already_correct_items = []

    """
    infile_dir = f"../fof/logs/{task}/fof_answer_rounds_1_feedback_rounds_3/few_shot/{model}/" # TODO
    test_example_cnt = 100
    with open(gen_feedback_file, 'r') as reader:
        feedback_items = [json.loads(l) for l in reader]
    for idx in range(test_example_cnt):
        with open(os.path.join(infile_dir, f"actor{idx}.jsonl"), 'r') as reader:
            jsonl_items = [json.loads(l) for l in reader]
        item = {
                "question" : jsonl_items[0]['content'].split('#')[1].replace("Q:", "").strip(),
                "answer" : "",
                "reference" : "",
                "wrong_code" : jsonl_items[1]['content'].replace("```", "").strip(),
                "corrections" : [""],
                "feedback" : feedback_items[idx]['predict'].replace("### END ###", "").strip(),
                "scores": [1],
                }
        if already_correct_message in item['feedback']:
            pseudo_generated_pred = {
                    "label" : "```\n\n```\n\n### END ###",
                    "predict" : item['wrong_code'], # skip refinement
                    "idx" : idx,
                    }
            already_correct_items.append(pseudo_generated_pred)
        else:
            items.append(item)
    """


feedback_items = []
no_errors_feedback_instructions = set()
refinement_items = []
feedback_refinement_items = []
for item in items:
    if 1 in item['scores']: # and wrong_code2score.get(item['wrong_code'], 0) == 0: # filter feedback and wrong_code == -1 (runtime error)
        item['feedback'] = item['feedback'].replace("Answer 1", "the provided code").replace("Answer 2", "the correct code")
        correct_code = item['corrections'][item['scores'].index(1)].replace("print(solution())", "").strip().replace("\n\n", "\n")
        if task == "gsm8k":
            feedback_item = {
                    "instruction" : f"# Q: {item['question']}\n# Can you answer this question by implementing a `solution()` Python function that returns the result? Please ensure to start and finish your code with ```. Note that your answer should only contain the code and nothing else.\n\n```\n{item['wrong_code']}\n```\n\n# There are errors in the code above because of lack of understanding of the question. What are the errors?\n",
                    "input" : "",
                    "output" : f"{item['feedback']}\n\n### END ###",
                    }
            refinement_item = {
                    "instruction" : f"# Q: {item['question']}\n# Can you answer this question by implementing a `solution()` Python function that returns the result? Please ensure to start and finish your code with ```. Note that your answer should only contain the code and nothing else.\n\n```\n{item['wrong_code']}\n```\n\n# There are errors in the code above because of lack of understanding of the question. What are the errors?\n\n{item['feedback']}\n\n# Can you correct the errors in the code? Please ensure to start and finish your code with ```. Note that your answer should only contain the code and nothing else.\n",
                    "input" : "",
                    "output" : f"```\n{correct_code}\n```\n\n### END ###",
                    }
            feedback_refinement_item = {
                    "instruction" : f"# Q: {item['question']}\n# Can you answer this question by implementing a `solution()` Python function that returns the result? Please ensure to start and finish your code with ```. Note that your answer should only contain the code and nothing else.\n```\n{item['wrong_code']}\n```\n\n# If there are any issues with the code provided, please identify these errors in the code.\n# If there are any issues with the code provided, please correct the errors in the code.\n",
                    "input" : "",
                    "output" : f"Feedback:\n\n{item['feedback']}\n\nCorrection:\n\n```\n{correct_code}\n```\n\n### END ###",
                    }
        elif task in ["gsm8k_nl", "csqa", "math", "riddlesense", "qasc"]:
            feedback_instruction = TASK_FEEDBACK_TYPE2FEEDBACK_INSTR[task][feedback_type]
            if "sft" in stage:
                if feedback_type == 0:
                    formatted_feedback = item['feedback']
                elif feedback_type == 1:
                    formatted_feedback = "NO"
                elif feedback_type == 2:
                    formatted_feedback = remove_copied_steps(item['feedback'])
                elif feedback_type == 3:
                    formatted_feedback = remove_copied_steps(item['feedback'], add_final_step_feedback=True)
                elif feedback_type == 4:
                    formatted_feedback = remove_copied_steps(item['feedback'], return_feedback_for_first_error_step=True)
                elif feedback_type == 5:
                    formatted_feedback = ""
                else:
                    raise ValueError(f"invalid feedback_type = {feedback_type}")
            else:
                formatted_feedback = item['feedback']

            feedback_item = {
                    "instruction" : f"Question: {item['question']}\n\nSolution:\n{item['wrong_code']}\n\n{feedback_instruction}",
                    "input" : "",
                    "output" : formatted_feedback,
                    }
            refinement_item = {
                    "instruction" : f"Question: {item['question']}\n\nSolution:\n{item['wrong_code']}\n\nFeedback:\n{formatted_feedback}\n\nPlease refine your reasoning based on the feedback. Your final answer should be a single numerical number, in the form \\boxed{{answer}}.\n\nCorrection:",
                    "input" : "",
                    "output" : correct_code,
                    }
            REFINER_FT_PROMPT = TASK2REFINER_FT_PROMPT[task]
            # f"Question: {item['question']}\n\nSolution:\n{item['wrong_code']}\n\n{feedback_instruction} Then refine your reasoning based on the feedback. Your final answer should be a single numerical number, in the form \\boxed{{answer}}.\n\nFeedback:",
            feedback_refinement_item = {
                    "instruction" : REFINER_FT_PROMPT.replace("{question}", item['question']).replace("{solution}", item['wrong_code']).replace("{feedback_instruction}", feedback_instruction),
                    "input" : "",
                    "output" : f"{formatted_feedback}\n\nCorrection:\n\n{correct_code}",
                    }
        else:
            raise ValueError(f"invalid task: {task}")

        feedback_items.append(feedback_item)
        refinement_items.append(refinement_item)
        feedback_refinement_items.append(feedback_refinement_item)
        if stage == "sft_rebalanced":
            no_errors_feedback_instruction = (item['question'], correct_code)
            # no_errors_feedback_instruction = f"# Q: {item['question']}\n# Can you answer this question by implementing a `solution()` Python function that returns the result? Please ensure to start and finish your code with ```. Note that your answer should only contain the code and nothing else.\n\n```\n{correct_code}\n```\n\n# There are errors in the code above because of lack of understanding of the question. What are the errors?\n"
            no_errors_feedback_instructions.add(no_errors_feedback_instruction)

if "sft" in stage:
    # deduplication
    print("pre-deduplication feedback_items cnt:", len(feedback_items))
    feedback_items = [dict(t) for t in {tuple(d.items()) for d in feedback_items}]
    refinement_items = [dict(t) for t in {tuple(d.items()) for d in refinement_items}]
    feedback_refinement_items = [dict(t) for t in {tuple(d.items()) for d in feedback_refinement_items}]
    print("post-deduplication feedback_items cnt:", len(feedback_items))


if stage == "sft_rebalanced":
    no_errors_feedback_instructions = list(no_errors_feedback_instructions)
    print("no_errors_feedback_instructions", len(no_errors_feedback_instructions))
    print(no_errors_feedback_instructions[0])
    # down-sample no_errors_feedback_cnt
    # has_error_feedback_cnt = len({item['instruction'] for item in feedback_items}) # only count unique input 
    has_error_feedback_cnt = len(feedback_items)
    no_errors_feedback_cnt = has_error_feedback_cnt * correct_generation_ratio / (1 - correct_generation_ratio)
    no_errors_feedback_cnt = int(no_errors_feedback_cnt)
    print("has_error_feedback_cnt", has_error_feedback_cnt)
    print("correct_generation_ratio", correct_generation_ratio)
    print("no_errors_feedback_cnt", no_errors_feedback_cnt)
    no_errors_feedback_instructions = random.sample(no_errors_feedback_instructions, k=no_errors_feedback_cnt)
    print("after sampling", no_errors_feedback_instructions[0])
    for instr in no_errors_feedback_instructions:
        item_question, correct_code = instr
        if task == "gsm8k":
            no_errors_feedback_item = {
                    "instruction" : f"# Q: {instr[0]}\n# Can you answer this question by implementing a `solution()` Python function that returns the result? Please ensure to start and finish your code with ```. Note that your answer should only contain the code and nothing else.\n\n```\n{instr[1]}\n```\n\n# There are errors in the code above because of lack of understanding of the question. What are the errors?\n",
                    "input" : "",
                    "output" : f"{already_correct_message}\n### END ###",
                    }
            feedback_items.append(no_errors_feedback_item)
            no_errors_feedback_refinement_item = {
                    "instruction" : f"# Q: {instr[0]}\n# Can you answer this question by implementing a `solution()` Python function that returns the result? Please ensure to start and finish your code with ```. Note that your answer should only contain the code and nothing else.\n# If there are any issues with the code provided, please identify these errors in the code.\n# If there are any issues with the code provided, please correct the errors in the code.\n",
                    "input" : "",
                    "output" : f"```\n{instr[1]}\n```\n\nFeedback:\n\n{already_correct_message}\n### END ###",
                    }
            feedback_refinement_items.append(no_errors_feedback_refinement_item)
        elif task == "gsm8k_nl":
            all_step_correct_feedback = gen_all_step_correct_feedback(correct_code)
            final_step_correct_feedback = re.findall("Step \d+", correct_code)[-1] + f": {already_correct_message}" 
            if feedback_type == 0:
                # original feedback
                no_errors_feedback_item = {
                        "instruction" : f"Question: {item_question}\n\nSolution:\n{correct_code}\n\nDoes the solution contains any errors? If yes, please go through each step in the solution, and provide feedback that helps correct the errors.",
                        "input" : "",
                        "output" : f"{all_step_correct_feedback}",
                        }
            elif feedback_type == 1:
                # 1 (binary classifier)
                no_errors_feedback_item = {
                        "instruction" : f"Question: {item_question}\n\nSolution:\n{correct_code}\n\nIs the solution correct?",
                        "input" : "",
                        "output" : f"YES",
                        }
            elif feedback_type == 2:
                # 2 (short feedback)
                no_errors_feedback_item = {
                        "instruction" : f"Question: {item_question}\n\nSolution:\n{correct_code}\n\nDoes the solution contains any errors? If yes, please go through each step in the solution, and provide feedback that helps correct the errors.",
                        "input" : "",
                        "output" : "This solution is correct.",
                        }
            elif feedback_type == 3:
                # 3 (no answer feedback)
                no_errors_feedback_item = {
                        "instruction" : f"Question: {item_question}\n\nSolution:\n{correct_code}\n\nDoes the solution contains any errors? If yes, please go through each step in the solution, and provide feedback that helps correct the errors.",
                        "input" : "",
                        "output" : re.findall("Step \d+", correct_code)[-1] + f": {already_correct_message}",
                        }
            elif feedback_type == 4:
                # 4 (first error step feedback)
                no_errors_feedback_item = {
                        "instruction" : f"Question: {item_question}\n\nSolution:\n{correct_code}\n\nDoes the solution contains any errors? If yes, please identify the first error step in the solution, and provide feedback that helps correct the error.",
                        "input" : "",
                        "output" : final_step_correct_feedback,
                        }
 

            feedback_items.append(no_errors_feedback_item)
            no_errors_feedback_refinement_item = {
                    "instruction" : f"Question: {item_question}\n\nSolution:\n{correct_code}\n\nDoes the solution contains any errors? If yes, please identify the first error step in the solution, and provide feedback that helps correct the error. Then refine your reasoning based on the feedback. Your final answer should be a single numerical number, in the form \\boxed{{answer}}.\n\nFeedback:",
                    "input" : "",
                    "output" : final_step_correct_feedback,
                    }
            feedback_refinement_items.append(no_errors_feedback_refinement_item)
    print("total feedbacks after rebalanced", len(feedback_items))

"""
if stage == "sft_balanced" or stage == "sft_rebalanced":
    no_errors_feedback_items = []
    no_errors_feedback_refinement_items = []
    # ft_rationale_items = []
    with open(f"logs/{task}/generations_{model}_score.jsonl", 'r') as reader:
        items = [json.loads(l) for l in reader]
        if "gsm8k" in task:
            items = items[:6500] # only use training set
        else:
            raise ValueError(f"invalid task: {task}")
    for item in items:
        for s, g in zip(item['score'][:SAMPLE_SIZE], item['generated_answers'][:SAMPLE_SIZE]):
            if s == 1:
                g = clean_cot(g) # .replace("print(solution())", "").strip()
                all_step_correct_feedback = gen_all_step_correct_feedback(g)
                if all_step_correct_feedback == "":
                    continue
                feedback_item = {
                        "instruction" : f"# Q: {item['question']}\n# Can you answer this question by implementing a `solution()` Python function that returns the result? Please ensure to start and finish your code with ```. Note that your answer should only contain the code and nothing else.\n\n```\n{g}\n```\n\n# There are errors in the code above because of lack of understanding of the question. What are the errors?\n", 
                        "input" : "",
                        "output" : f"{already_correct_message}\n### END ###",
                        }
                feedback_refinement_item = {
                        "instruction" : f"Question: {item['question']}\n\nSolution:\n{g}\n\nDoes the solution contains any errors? If yes, please go through each step in the solution, and provide feedback that helps correct the errors. Then refine your reasoning based on the feedback. Your final answer should be a single numerical number, in the form \\boxed{{answer}}.",
                    "input" : "",
                    "output" : f"This solution is correct.",
                    }

                ft_rationale_item = {
                        "instruction" : f"# Q: {item['question']}\n# Can you answer this question by implementing a `solution()` Python function that returns the result? Please ensure to start and finish your code with ```. Note that your answer should only contain the code and nothing else.\n",
                        "input" : "",
                        "output" : f"\n```\n{g}\n```\n\n### END ###",
                        }
                ft_rationale_items.append(ft_rationale_item)
                no_errors_feedback_items.append(feedback_item)
                no_errors_feedback_refinement_items.append(feedback_refinement_item)
    print(f"{len(no_errors_feedback_items)} no error feedback / ft_rationale / feedback_refinement items")
    print(f"{len(feedback_items)} error correcting feedback")
    # feedback_items = no_errors_feedback_items # DEBUG
    if stage == "sft_rebalanced":
        # down-sample no_errors_feedback_cnt
        correct_generation_ratio = get_correct_generation_ratio(task, model)
        has_error_feedback_cnt = len(feedback_items)
        no_errors_feedback_cnt = has_error_feedback_cnt * correct_generation_ratio / (1 - correct_generation_ratio)
        no_errors_feedback_cnt = int(no_errors_feedback_cnt)
        print("has_error_feedback_cnt", has_error_feedback_cnt)
        print("correct_generation_ratio", correct_generation_ratio)
        print("no_errors_feedback_cnt", no_errors_feedback_cnt)
        no_errors_feedback_idx = random.sample(list(range(len(no_errors_feedback_items))), k=no_errors_feedback_cnt)
        no_errors_feedback_items = [no_errors_feedback_items[idx] for idx in no_errors_feedback_idx]
        no_errors_feedback_refinement_items = [no_errors_feedback_refinement_items[idx] for idx in no_errors_feedback_idx]

    feedback_items.extend(no_errors_feedback_items)
    feedback_refinement_items.extend(no_errors_feedback_refinement_items)

    outfile_ft_rationale = f"../LLaMA-Factory/data/{task}_ft_rationale_{model}.json"
    with open(outfile_ft_rationale, 'w') as writer:
        json.dump(ft_rationale_items, writer, indent=4)
    print(outfile_ft_rationale)
"""

if stage in ["predict_balanced_refinement", "predict_rebalanced_refinement", "predict_cot_ft_as_input_rebalanced_refinement"]:
    # save already-correct generations
    outfile_already_correct = outfile_refinement.replace("data", "predictions").replace(".json", "") 
    """
    if stage == "predict_refinement":
        outfile_already_correct = f"../LLaMA-Factory/predictions/{task}_test_refinement_{model}"
    elif stage == "predict_balanced_refinement":
        outfile_already_correct = f"../LLaMA-Factory/predictions/{task}_test_balanced_refinement_{model}"
    elif stage == "predict_rebalanced_refinement":
        outfile_already_correct = outfile_refinement.replace("data", "predictions").replace(".json", "") 
    elif stage == "predict_cot_ft_as_input_rebalanced_refinement":
        outfile_already_correct = f"../LLaMA-Factory/predictions/{task}_{split}_rebalanced_refinement_{model}"
    """


    os.makedirs(outfile_already_correct, exist_ok=True)
    outfile_already_correct = os.path.join(outfile_already_correct, "already_correct.jsonl") 
    with open(outfile_already_correct, 'w') as writer:
        for item in already_correct_items:
            writer.write(json.dumps(item)+'\n')
    print(outfile_already_correct, f"has {len(already_correct_items)} items")

"""
with open(outfile_feedback, 'w') as writer: 
    json.dump(feedback_items, writer, indent=4)
print(f"{len(feedback_items)} items in", outfile_feedback)

with open(outfile_refinement, 'w') as writer: 
    json.dump(refinement_items, writer, indent=4)
print(f"{len(refinement_items)} items in", outfile_refinement)

with open(outfile_feedback_refinement, 'w') as writer: 
    json.dump(feedback_refinement_items, writer, indent=4)
print(f"{len(feedback_refinement_items)} items in", outfile_feedback_refinement)
"""

save_to_file(feedback_items, outfile_feedback, chunk)
save_to_file(refinement_items, outfile_refinement, chunk)
save_to_file(feedback_refinement_items, outfile_feedback_refinement, chunk)
