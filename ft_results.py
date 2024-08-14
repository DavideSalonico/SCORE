import json
import os
import re
import numpy as np
import argparse

from tqdm import tqdm
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix

from pal_code_exec import compute
from util import *
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str, choices=TASK_LIST)
parser.add_argument("--stage", type=str, choices=['choose_best_rft_checkpoint', 'eval_greedy_refine', 'eval_sampling_refine'])
parser.add_argument("--input", type=str, choices=['prompted', 'cot_ft'], help='source of intial generations')
parser.add_argument("--input_model", type=str, choices=MODEL_LIST, default='meta-llama/Llama-2-13b-chat-hf')
parser.add_argument("--verifier_type", type=str, choices=['prompted', 'ours_ft_on_prompted_solutions', 'ours_ft_on_cot_ft_solutions', 'ours_ft_cot_init_on_prompted_solutions', 'ours_ft_base_init_on_cot_ft_solutions', 'ours_ft_on_base_rft_solutions', 'ours_ft_base_init_on_base_rft_solutions'])
parser.add_argument("--verifier_model", type=str, choices=MODEL_LIST+['gpt-35-turbo', 'gpt-35-turbo-sc', 'gpt-4'])
parser.add_argument("--refiner_type", type=str, choices=['prompted', 'filtered_feedback4_refinement_diff', 'filtered_feedback4_refinement_cot_ft_diff', 'filtered_feedback4_refinement_base_rft_diff'])
parser.add_argument("--refiner_model", type=str, choices=MODEL_LIST, default='meta-llama/Llama-2-13b-chat-hf')
parser.add_argument("--refiner_ft_data_size", type=str, choices=["3625", "7250", "10875", "more"], default='more')
parser.add_argument("--refiner_cot_init", action='store_true')
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--split", type=str, choices=["dev", "test"])
parser.add_argument("--best_dev_confidence", type=float, default=None)

args = parser.parse_args()

TASK = args.task
split = args.split
stage = args.stage
INPUT = args.input
INPUT_MODEL = args.input_model.replace('/', '-')
VERIFIER_TYPE = args.verifier_type
VERIFIER_MODEL = args.verifier_model.replace('/', '-')
REFINER_TYPE = args.refiner_type
REFINER_MODEL = args.refiner_model.replace('/', '-')
VERBOSE = args.verbose
best_dev_confidence = args.best_dev_confidence
refiner_cot_init = args.refiner_cot_init
refiner_ft_data_size = args.refiner_ft_data_size


already_correct_message = TASK2ALREADY_CORRECT_MSG[TASK]
parsing_func = TASK2PARSING_FUNC[TASK]
dev_size = TASK2DEV_SIZE[TASK] 

def clean_code(c):
    c_findall = re.findall("\`\`\`\s*(.*?)\`\`\`", c, re.DOTALL)
    if len(c_findall) == 0:
        code_part = c
    else:
        code_part = c_findall[-1] # correction or already-correct generation
    return code_part.replace("```", "").replace("### END ###", "").strip() + "\nprint(solution())"

if stage == "eval_greedy_refine":
    if split == "dev":
        initial_prompting_solution_infile = f"logs/{TASK}/generations_{INPUT_MODEL}_score.jsonl_greedy"
    elif split == "test":
        initial_prompting_solution_infile = f"logs/{TASK}/generations_{INPUT_MODEL}_score_test.jsonl_greedy"
    with open(initial_prompting_solution_infile, 'r') as reader:
        items = [json.loads(l) for l in reader]
        if split == "dev" and len(items) > dev_size:
            items = items[dev_size:]
        initial_prompting_solution_scores = [item['score'][0] for item in items]

if stage == "eval_sampling_refine":
    if split == "dev":
        initial_prompting_sampling_solution_infile = f"logs/{TASK}/generations_{INPUT_MODEL}_score.jsonl"
    elif split == "test":
        initial_prompting_sampling_solution_infile = f"logs/{TASK}/generations_{INPUT_MODEL}_score_test.jsonl"
    with open(initial_prompting_sampling_solution_infile, 'r') as reader:
        items = [json.loads(l) for l in reader]
        if split == "dev" and len(items) > dev_size:
            items = items[dev_size:]
        initial_prompting_sampling_solution_scores = [majority_voting(item['extracted_answers'], item['reference'])[-1] for item in items]

if split == "dev":
    refs_infile = f"data/{TASK}/train_shuf.jsonl"
elif split == "test":
    refs_infile = f"data/{TASK}/test_shuf.jsonl"
with open(refs_infile, 'r') as reader:
    if TASK in ["gsm8k", "gsm8k_nl", "math"]:
        refs = [json.loads(l)['answer'].split('### ')[-1] for l in reader]
    elif TASK in ["csqa", "riddlesense", "qasc"]:
        refs = [json.loads(l)['answer'] for l in reader]
    else:
        raise ValueError(f"invalid TASK name {TASK}")
    if split == "dev":
        refs = refs[dev_size:]


test_example_cnt = len(refs)
# pred_dir = "../LLaMA-Factory/predictions/gsm8k_filtered_rebalanced_feedback_refinement_diff_meta-llama-Llama-2-13b-chat-hf" 
# pred_dir = "../LLaMA-Factory/predictions/gsm8k_test_rebalanced_refinement_meta-llama-Llama-2-13b-chat-hf" 
# pred_dir = "../LLaMA-Factory/predictions/gsm8k_ft_rationale_meta-llama-Llama-2-7b-chat-hf/"
def get_scores(pred_dir, return_preds=False):
    if "jsonl" not in pred_dir:
        pred_file = os.path.join(pred_dir, "generated_predictions.jsonl.00")
        if not os.path.exists(pred_file):
            pred_file = pred_file.replace(".00", "")
            if not os.path.exists(pred_file):
                pred_file = pred_file + "_greedy"
    else:
        pred_file = pred_dir
    # pred_file = "../LLaMA-Factory/data/gsm8k_test_feedback_meta-llama-Llama-2-13b-chat-hf.json"
    with open(pred_file, "r") as reader:
        refined_predictions = [clean_cot(json.loads(l)['predict']) for l in reader]
        # refined_predictions = [l['instruction'].replace("```", "") + "\nprint(solution())" for l in json.load(reader)]
        # print(refined_predictions[:3])

    already_correct_file = os.path.join(pred_dir, "already_correct.jsonl")
    if os.path.exists(already_correct_file):
        with open(already_correct_file, 'r') as reader:
            items = [json.loads(l) for l in reader]
            already_correct_predictions = [clean_code(item['predict']) for item in items]
            already_correct_idx = [item['idx'] for item in items]

        # print(f"{len(refined_predictions)} refined_predictions, {len(already_correct_predictions)} already_correct_predictions")
        # print(already_correct_idx)
        predictions = [] 
        for idx in range(test_example_cnt):
            if len(already_correct_idx) and idx == already_correct_idx[0]:
                already_correct_idx.pop(0)
                predictions.append(already_correct_predictions.pop(0))
            else:
                predictions.append(refined_predictions.pop(0))
    else:
        predictions = refined_predictions


    # with open("data/gsm8k/test_shuf.jsonl", 'r') as reader:
    #     refs = [json.loads(l)['answer'].split('#### ')[-1] for l in reader]
    #     predictions = predictions[:test_example_cnt]
    # predictions = [[pred] for pred in predictions]
    # refs = refs[:test_example_cnt]

    # scores = compute(predictions, refs)
    """
    print(predictions)
    print(refs)
    print(scores)
    """
    # scores = [s[0] for s in scores]
    if return_preds:
        return [parsing_func(pred) for pred in predictions]
    else:
        assert len(predictions) == len(refs)
        scores = [int(is_same(parsing_func(pred), ref)) for pred, ref in zip(predictions, refs)]
        correct = len([s for s in scores if s == 1])
        accu = correct * 100 / len(scores)
        print(pred_file)
        print(f"accu", accu)
        
        """
        # 0 for refine (initial solution incorrect), 1 for copy intital solution
        pred_when_to_sc = [int(already_correct_message in pred) for pred in predictions]
        if "cot_ft_as_input" not in pred_file:
            # only support greedy output now
            gold_when_to_sc = initial_prompting_solution_scores
        else:
            raise ValueError("get_scores() and initial_cot_scores reply on each other, should decompose get_scores into two functions")
        assert len(pred_when_to_sc) == len(gold_when_to_sc)
        
        sc_contribution = sum([1 for p, c in zip(pred_when_to_sc, scores) if p == 0 and c == 1]) / sum([1 for p in pred_when_to_sc if p == 0])

        sc_freq = 1 - sum(pred_when_to_sc) / len(pred_when_to_sc)
        print("When to self-correct")
        print("initial correctness -> critic (determine when to self-correct) correctness")
        print(f"Self-correct Frequency: {100*sc_freq:.2f}%")
        print(f"Self-correct Contribution: {100*sc_contribution:.2f}%")
        print(classification_report(gold_when_to_sc, pred_when_to_sc))
        print(confusion_matrix(gold_when_to_sc, pred_when_to_sc))
        print("initial correctness -> refined correctness")
        print(confusion_matrix(initial_prompting_solution_scores, scores))
        """

        """
        if "feeback_refinement" not in pred_file: # separate critic refiner acrh
            gen_feedback_file = pred_file.replace("_refinement", "")
        else:
            gen_feedback_file = pred_file

        with open(gen_feedback_file, 'r') as reader:
            gen_feedbacks = [json.loads(l)['predict']) for l in reader]
        """


        return scores

def get_sampling_evals(dataset_model, SAMPLE_SIZE=10):
    # SAMPLE_SIZE 
    # return (prediction, confidence, correctness)
    if dataset_model in ["dev_rationale_train_rationale", "dev_feedback4_refinement_filtered_feedback4_refinement_diff"]:
        sampling_preds = [s for sample_idx in range(SAMPLE_SIZE) for part_idx in range(16) for s in get_scores(f"../LLaMA-Factory/predictions/more_{TASK}_part{part_idx:02}_{dataset_model}_meta-llama-Llama-2-13b-chat-hf/generated_predictions.jsonl.{sample_idx:02}", return_preds=True)]
    else:
        sampling_preds = [s for i in range(16) for s in get_scores(f"../LLaMA-Factory/predictions/{TASK}_part{i:02}_{dataset_model}_meta-llama-Llama-2-13b-chat-hf/generated_predictions.jsonl.00", return_preds=True)]
    assert len(sampling_preds) == len(refs) * SAMPLE_SIZE, f"len(sampling_preds)={len(sampling_preds)}, len(refs)={len(refs)}, SAMPLE_SIZE={SAMPLE_SIZE}"
    sampling_preds = [sampling_preds[i:i+len(refs)] for i in range(0, len(sampling_preds), len(refs))]
    assert len(sampling_preds) == SAMPLE_SIZE
    sampling_preds = [[one_version[data_idx] for one_version in sampling_preds] for data_idx in range(len(refs))]
    assert len(sampling_preds) == len(refs)
    sampling_evals = [majority_voting(preds, ref)[-1] for preds, ref in zip(sampling_preds, refs)]
    return sampling_evals


def get_oracle_scores(initial_scores, refined_scores):
    assert len(initial_scores) == len(refined_scores)
    oracle_scores = [int(ini or s) for ini, s in zip(initial_scores, refined_scores)]
    print(f"oracle refinement accu: {100*sum(oracle_scores)/len(oracle_scores):.2f}")



def get_verifier_scores(original_verifier_scores, initial_scores, refined_scores, confidence=None, detailed_report=False):
    if confidence is None:
        all_confidence = [round(i, 2) for i in np.linspace(0.5,1,51).tolist()]
    else:
        all_confidence = [confidence]
    assert len(original_verifier_scores) == len(initial_scores)
    assert len(initial_scores) == len(refined_scores)
    best_final_accu = 0
    best_conf = None
    for conf in all_confidence:
        if type(original_verifier_scores[0]) is int:
            verifier_scores = original_verifier_scores
        else:
            verifier_scores = [0 if v['predict'] == 0 and v['confidence'] >= conf else 1 for v in original_verifier_scores]
        system_scores = [ini if v else r for v, ini, r in zip(verifier_scores, initial_scores, refined_scores)]
        final_accu_at_conf = 100*sum(system_scores)/len(system_scores)
        if final_accu_at_conf > best_final_accu:
            best_final_accu = final_accu_at_conf
            best_conf = conf
        # if detailed_report:
        #     print(f"ours verifier (confidence @ {conf:.2f}) accu: {final_accu_at_conf:.2f}")
    print(f"ours verifier (confidence @ {best_conf:.2f}) accu: {best_final_accu:.2f}")

    if not detailed_report:
        return
    print(f"For Best confidence ({best_conf:.2f})")
    if type(original_verifier_scores[0]) is int:
        verifier_scores = original_verifier_scores
    else:
        verifier_scores = [0 if v['predict'] == 0 and v['confidence'] > best_conf else 1 for v in original_verifier_scores]
    system_scores = [ini if v else r for v, ini, r in zip(verifier_scores, initial_scores, refined_scores)]

    try:
        sc_contribution = len([r for v, r in zip(verifier_scores, refined_scores) if v == 0 and r == 1]) / len([v for v in verifier_scores if v == 0])
    except:
        sc_contribution = 0
    sc_freq = len([v for v in verifier_scores if v == 0]) / len(verifier_scores)
    print("When to self-correct")
    print("initial correctness -> critic (determine when to self-correct) correctness")
    print(f"Self-correct Frequency: {100*sc_freq:.2f}%")
    print(f"Self-correct Contribution: {100*sc_contribution:.2f}%")
    print(classification_report(initial_scores, verifier_scores, digits=3))
    print(confusion_matrix(initial_scores, verifier_scores))
    print("initial correctness -> refined correctness")
    print(confusion_matrix(initial_scores, system_scores))


if stage == "choose_best_rft_checkpoint":
    for pred_dir in [f"../LLaMA-Factory/predictions/{TASK}_dev_rationale_train_rationale_meta-llama-Llama-2-13b-chat-hf_step_{ckpt}" for ckpt in ["3309", "6618", "9927"]]:
        get_scores(pred_dir)
    exit(0)

# prompted_as_input_scores = get_scores(f"../LLaMA-Factory/predictions/{TASK}_dev_feedback_refinement_filtered_rebalanced_feedback_refinement_diff_meta-llama-Llama-2-13b-chat-hf")
# get_oracle_scores(initial_prompting_solution_scores, prompted_as_input_scores)
"""
assert len(initial_prompting_solution_scores) == len(prompted_as_input_scores)
oracle_scores = [int(ini or s) for ini, s in zip(initial_prompting_solution_scores, prompted_as_input_scores)]
print(f"oracle refinement accu: {100*sum(oracle_scores)/len(oracle_scores):.2f}")
"""

if VERIFIER_TYPE == "ours_ft_on_prompted_solutions":
    if split == "dev":
        verifier_pred_file = f"checkpoint/verifier/{TASK}/{VERIFIER_MODEL}/dev.json"
    elif split == "test":
        verifier_pred_file = f"checkpoint/verifier/{TASK}/{VERIFIER_MODEL}/test.json" 
elif VERIFIER_TYPE == "prompted":
    verifier_pred_file = f"checkpoint/verifier/{TASK}/cot_ft_train_rationale_{INPUT_MODEL}_verified_by_{VERIFIER_MODEL}/generated_predictions_test.jsonl_greedy" 
    if INPUT == "prompted":
        verifier_pred_file = verifier_pred_file.replace("cot_ft_train_rationale_", "")
    if split == "dev":
        verifier_pred_file = verifier_pred_file.replace("_test", "")
    if "-sc" in VERIFIER_MODEL:
        verifier_pred_file = verifier_pred_file.replace("_greedy", "").replace("-sc", "")
elif VERIFIER_TYPE == "ours_ft_on_cot_ft_solutions":
    if split == "dev":
        verifier_pred_file = f"checkpoint/verifier/{TASK}/cot_ft_train_rationale_{VERIFIER_MODEL}/generated_predictions.jsonl"
    elif split == "test":
        verifier_pred_file = f"predictions/verifier/{TASK}/cot_ft_train_rationale_{VERIFIER_MODEL}/test_cot_ft.json"
elif VERIFIER_TYPE == "ours_ft_cot_init_on_prompted_solutions":
    if split == "dev":
        verifier_pred_file = f"checkpoint/verifier/{TASK}/train_rationale_{VERIFIER_MODEL}/generated_predictions.jsonl"
    elif split == "test":
        verifier_pred_file = f"predictions/verifier/{TASK}/{VERIFIER_MODEL}/test.json" 
elif VERIFIER_TYPE == "ours_ft_base_init_on_cot_ft_solutions":
    if split == "dev":
        verifier_pred_file = f"checkpoint/verifier/{TASK}/cot_ft_{VERIFIER_MODEL}/generated_predictions.jsonl"
    elif split == "test":
        assert 0, "only for dev ablation"
elif VERIFIER_TYPE in ["ours_ft_on_base_rft_solutions", "ours_ft_base_init_on_base_rft_solutions"]:
    if INPUT == "prompted":
        verifier_pred_file = f"predictions/verifier/{TASK}/base_rft_train_rationale_{VERIFIER_MODEL}/{split}.json"
    elif INPUT == "cot_ft":
        verifier_pred_file = f"predictions/verifier/{TASK}/base_rft_train_rationale_{VERIFIER_MODEL}/{split}_cot_ft.json"
    if "base_init" in VERIFIER_TYPE:
        verifier_pred_file = verifier_pred_file.replace("_train_rationale", "")
else:
    raise ValueError(f"invalid VERIFIER_TYPE: {VERIFIER_TYPE}")
# verifier_pred_file = f"checkpoint/verifier/{TASK}/cot_ft_train_rationale_meta-llama-Llama-2-13b-chat-hf_verified_by_gpt-4/generated_predictions.jsonl_greedy"
with open(verifier_pred_file, 'r') as reader:
    initial_prompting_verifier_scores = [json.loads(l) for l in reader] 
    assert len(initial_prompting_verifier_scores) == len(initial_prompting_solution_scores)

if INPUT == "prompted":
    initial_scores = initial_prompting_solution_scores
    print(f"{split} initial prompting solution accu:", sum([s for s in initial_prompting_solution_scores if s == 1]) / len(initial_prompting_solution_scores))
elif INPUT == "cot_ft":
    initial_cot_indir = f"../LLaMA-Factory/predictions/{TASK}_{split}_rationale_train_rationale_{INPUT_MODEL}/"
    initial_cot_scores = get_scores(initial_cot_indir)
    initial_scores = initial_cot_scores
    print(f"{split} initial cot-finetuned solution accu:", sum([s for s in initial_cot_scores if s == 1]) / len(initial_cot_scores))
else:
    raise ValueError(f"invalid INPUT: {INPUT}")

"""
print('='*30, "feedback4_separate_refine")
feedback4_separate_refine_scores = get_scores(f"../LLaMA-Factory/predictions/{TASK}_dev_feedback4_filtered_feedback4_diff_refinement_meta-llama-Llama-2-13b-chat-hf")
get_oracle_scores(initial_scores, feedback4_separate_refine_scores)
get_verifier_scores(initial_prompting_verifier_scores, initial_scores, feedback4_separate_refine_scores, detailed_report=False)
"""

if stage == "eval_greedy_refine":
    print('='*30, "feedback4_combined_critic_refine")
    if REFINER_TYPE == "prompted":
        refine_infile = f"predictions/{TASK}_{split}_feedback4_refinement_1samples_self_refine_{REFINER_MODEL}/generated_predictions_test.jsonl_greedy" 
    else:
        if INPUT == "prompted": 
            refine_infile = f"../LLaMA-Factory/predictions/more_{TASK}_{split}_feedback4_refinement_1samples_{REFINER_TYPE}_{REFINER_MODEL}"
        elif INPUT == "cot_ft":
            refine_infile = f"../LLaMA-Factory/predictions/more_{TASK}_{split}_cot_ft_as_input_feedback4_refinement_1samples_{REFINER_TYPE}_{REFINER_MODEL}"
        if refiner_cot_init:
            refine_infile = refine_infile.replace("more", "more_cot_init")
        refine_infile = refine_infile.replace("more", refiner_ft_data_size)
        if not os.path.exists(refine_infile):
            refine_infile = refine_infile.replace("_1samples", "")
    feedback4_combined_critic_refine_scores = get_scores(refine_infile)
    # f"../LLaMA-Factory/predictions/more_{TASK}_dev_feedback4_refinement_1samples_filtered_feedback4_refinement_base_rft_diff_meta-llama-Llama-2-13b-chat-hf")
    get_oracle_scores(initial_scores, feedback4_combined_critic_refine_scores)
    get_verifier_scores(initial_prompting_verifier_scores, initial_scores, feedback4_combined_critic_refine_scores, detailed_report=VERBOSE, confidence=best_dev_confidence)
    print('='*30, "oracle verifier")
    get_verifier_scores(initial_scores, initial_scores, feedback4_combined_critic_refine_scores, detailed_report=VERBOSE, confidence=1.0)
    exit(0)

"""
print('='*30, "feedback4_combined_critic_refine_rebalanced_0.13")
feedback4_combined_critic_refine_rebalanced_13_scores = get_scores(f"../LLaMA-Factory/predictions/{TASK}_dev_feedback_refinement_filtered_rebalanced_feedback_refinement_diff_meta-llama-Llama-2-13b-chat-hf")
get_oracle_scores(initial_prompting_solution_scores, feedback4_combined_critic_refine_rebalanced_13_scores)

print('='*30, "feedback3_separate_rebalanced_0.13")
feedback3_separate_rebalanced_13_scores = get_scores(f"../LLaMA-Factory/predictions/{TASK}_dev_feedback3_filtered_rebalanced_feedback3_0.13_diff_refinement_meta-llama-Llama-2-13b-chat-hf")
get_oracle_scores(initial_prompting_solution_scores, feedback3_separate_rebalanced_13_scores)
"""

"""
cot_as_input_scores = get_scores("../LLaMA-Factory/predictions/{TASK}_dev_cot_ft_as_input_feedback_refinement_filtered_rebalanced_feedback_refinement_diff_meta-llama-Llama-2-13b-chat-hf")
initial_cot_scores = get_scores(f"../LLaMA-Factory/predictions/{TASK}_dev_rationale_meta-llama-Llama-2-13b-chat-hf_rank32_lr5e-5_2337")
get_oracle_scores(initial_cot_scores, cot_as_input_scores)
"""

"""
# self-consistency
sampling_cot_as_input_refined_evals = get_evals("dev_cot_ft_as_input_feedback_refinement_filtered_feedback_refinement_diff")
sampling_cot_as_input_initial_evals = get_evals("dev_rationale_train_rationale")
initial_cot_sc_accu = sum([correctness for _, _, correctness in sampling_cot_as_input_initial_evals]) / len(refs)
print(f"initial_cot_sc_accu: {100*initial_cot_sc_accu:.2f}")
for confidence_threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    refined_cot_sc_accu = sum([init_correctness if init_confidence >= confidence_threshold else refined_correctness for (_, init_confidence, init_correctness), (_, _, refined_correctness) in zip(sampling_cot_as_input_initial_evals, sampling_cot_as_input_refined_evals)]) / len(refs)
    print(f"refined_cot_sc_accu @ {confidence_threshold}: {100*refined_cot_sc_accu:.2f}")
oracle_refined_cot_sc_accu = sum([int(init_correctness or refined_correctness) for (_, init_confidence, init_correctness), (_, _, refined_correctness) in zip(sampling_cot_as_input_initial_evals, sampling_cot_as_input_refined_evals)]) / len(refs)
print(f"oracle_refined_cot_sc_accu: {100*oracle_refined_cot_sc_accu:.2f}")
"""

if stage == "eval_sampling_refine":
    print("dev initial prompting sampling@10 solution accu:", sum([s for s in initial_prompting_sampling_solution_scores if s == 1]) / len(initial_prompting_sampling_solution_scores))
    print('='*30, "feedback4_combined_critic_refine, Q->1*A->10*A'")
    sampling_refinement_evals = get_sampling_evals("dev_feedback4_refinement_filtered_feedback4_refinement_diff")
    get_oracle_scores(initial_scores, sampling_refinement_evals)
    get_verifier_scores(initial_prompting_verifier_scores, initial_scores, sampling_refinement_evals, detailed_report=False)


"""
for dtst in ["dev_feedback4"]:
    for ratio in [0.11, 0.12, 0.13, 0.14]:
        get_scores(f"../LLaMA-Factory/predictions/{TASK}_{dtst}_filtered_rebalanced_feedback4_{ratio}_diff_refinement_meta-llama-Llama-2-13b-chat-hf")
"""
