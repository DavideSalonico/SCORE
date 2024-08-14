import os
import warnings
from collections import Counter, defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

from python_executor import run_program
from util import *
# adapted from https://github.com/huggingface/evaluate/blob/main/metrics/code_eval/code_eval.py

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval" metric executes untrusted model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions,
set the environment variable HF_ALLOW_CODE_EVAL="1". Within Python you can to this
with:
>>> import os
>>> os.environ["HF_ALLOW_CODE_EVAL"] = "1"
################################################################################\
"""


def compute(
    predictions,
    references,
    num_workers=4,
    timeout=3.0,
    majority_voting=False,
    answer_symbol=None,
    return_answers=False,
):
    """
    Returns the scores

    :param majority_voting: bool
        Takes majority voted answer to evaluate against the reference , defaults to False

    :param answer_symbol: str
        If speficifed the result of execution is fetched from the program's global context,
        the program is expected to have the variable name mentioned in `answer_symbol` that is available in globals.
        if not specified, the result are fetched from the stdout of the execution
        defaults to None.

    """

    if os.getenv("HF_ALLOW_CODE_EVAL", 0) != "1":
        raise ValueError(_WARNING)

    if os.name == "nt":
        raise NotImplementedError("This metric is currently not supported on Windows.")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        for task_id, candidates in enumerate(tqdm(predictions)):
            for candidate in candidates:
                args = (candidate, timeout, task_id, completion_id[task_id])
                if answer_symbol:
                    args += (answer_symbol,)
                future = executor.submit(run_program, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

        for future in as_completed(futures):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    answers = [None] * len(results)
    for result in results.values():
        result.sort()
        task_id = result[0][1]["task_id"]
        # filtering the failed generations to avoid influencing majority voting
        eval_answers = [
            r[1]["result"]
            for r in result
            # if isinstance(r[1]["result"], str)
            # and not r[1]["result"].startswith("failed:")
        ]
        # if all generations are failed - default to empty str for soring
        eval_answers = [""] if len(eval_answers) == 0 else eval_answers
        if majority_voting:
            counter = Counter(eval_answers)
            eval_answers = [counter.most_common()[0][0]]

        # if not majority_voting and len(eval_answers) > 1:
        #     warnings.warn(
        #         f"Multiple generations found for a task without setting `majority_voting` to True, defaulting answers from first generation"
        #     )
        # answers[task_id] = eval_answers[0]
        # return eval results of all generated answers
        answers[task_id] = eval_answers

    scores = []
    # Number of code generated that failed execution.
    errored = 0
    # print("DEBUG answers:", answers)
    # print("DEBUG references:", references)
    for task_id, (ans, ref) in enumerate(zip(answers, references)):
        score = []
        for a in ans:
            try:
                s = 1 if abs(float(a) - float(ref)) < 1e-3 else 0
            except ValueError as e:
                errored += 1
                s = -1 # syntax error as -1
                # print("error answer:", a)
            score.append(s)

        scores.append(score)
    if return_answers:
        return scores, answers
    else:
        return scores
    # return {"accuracy": sum(scores) / len(scores), "num_failed_execution": errored}

if __name__ == "__main__":
    import json
    import os
    import argparse

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--task", type=str, choices=TASK_LIST)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--split", type=str, default="train", choices=["train", "dev", "test", "train_cot_ft", "dev_cot_ft", "test_cot_ft", "dev_refinement", "test_refinement"])
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--greedy", action='store_true')
    args = parser.parse_args()
    TASK = args.task
    split = args.split

    RANGE = TASK2RANGE[TASK]
    TEST_RANGE = TASK2TEST_RANGE[TASK]
    dev_size = TASK2DEV_SIZE[TASK]
    INTERVAL = TASK2INTERVAL[TASK]
    deployment_id = args.model.replace('/', '-')
    use_greedy_decoding = args.greedy
    version = args.version

    items = []
    if split == "train" or split == "dev":
        for start_idx in RANGE:
            if split == "dev" and start_idx < dev_size:
                continue
            infile = f"logs/{TASK}/generations_{deployment_id}_{start_idx}_{start_idx+INTERVAL}.jsonl"
            if use_greedy_decoding:
                infile += "_greedy"
            if split == "dev" and not os.path.exists(infile):
                continue
            with open(infile, 'r') as reader:
                version_zero_items = [json.loads(l) for l in reader]

            infiles = []
            if version != 0:
                for v in range(1, version+1):
                    infiles.append(infile+f".version{v}")
            for infile in infiles:
                with open(infile, 'r') as reader:
                    extra_items = [json.loads(l) for l in reader]
                    assert len(version_zero_items) == len(extra_items), f"{infile} wrong items cnt, len(version_zero_items) = {len(version_zero_items)}, len(extra_items) = {len(extra_items)}"
                for version_zero_item, extra_item in zip(version_zero_items, extra_items):
                    version_zero_item["generated_answers"].extend(extra_item['generated_answers'])
            items.extend(version_zero_items)
    elif split == "test":
        for start_idx in TEST_RANGE:
            infile = f"logs/{TASK}/generations_{deployment_id}_{start_idx}_{start_idx+INTERVAL}_test.jsonl"
            if use_greedy_decoding:
                infile += "_greedy"

            with open(infile, 'r') as reader:
                items.extend([json.loads(l) for l in reader])
    elif split == "train_cot_ft" or split == "test_refinement" or split == "dev_refinement":
        if "train" in split or "dev" in split:
            with open(f"data/{TASK}/train_shuf.jsonl", 'r') as reader:
                items = [json.loads(l) for l in reader]
                if "train" in split:
                    items = items[:dev_size]
                elif "dev" in split:
                    items = items[dev_size:]
        elif "test" in split:
            with open(f"data/{TASK}/test_shuf.jsonl", 'r') as reader:
                items = [json.loads(l) for l in reader]
        for item in items:
            item['generated_answers'] = []
        for sample in range(1):
            item_idx = 0
            for part in range(16):
                if split == "train_cot_ft":
                    infile = f"../LLaMA-Factory/predictions/more_{TASK}_part{part:02}_train_for_inference_rationale_train_rationale_{deployment_id}/generated_predictions.jsonl.{sample:02}"
                elif split == "test_refinement":
                    infile = f"../LLaMA-Factory/predictions/more_{TASK}_part{part:02}_test_feedback4_refinement_10samples_filtered_feedback4_refinement_diff_{deployment_id}/generated_predictions.jsonl"
                elif split == "dev_refinement":
                    infile = f"../LLaMA-Factory/predictions/more_{TASK}_part{part:02}_dev_feedback4_refinement_10samples_filtered_feedback4_refinement_diff_{deployment_id}/generated_predictions.jsonl"


                with open(infile, 'r') as reader:
                    for l in reader:
                        pred_item = json.loads(l)
                        items[item_idx%len(items)]['generated_answers'].append(pred_item['predict'])
                        item_idx += 1
            assert item_idx == 10 * len(items), f"item_idx = {item_idx}, while len(items) = {len(items)}"
    elif split == "dev_cot_ft" or split == "test_cot_ft":
        if "dev" in split:
            with open(f"data/{TASK}/train_shuf.jsonl", 'r') as reader:
                items = [json.loads(l) for l in reader][dev_size:]
        elif "test" in split:
            with open(f"data/{TASK}/test_shuf.jsonl", 'r') as reader:
                items = [json.loads(l) for l in reader]
        for item in items:
            item['generated_answers'] = []

        with open(f"../LLaMA-Factory/predictions/{TASK}_{split.replace('_cot_ft', '')}_rationale_train_rationale_{deployment_id}/generated_predictions.jsonl.00", 'r') as reader:
            for item_idx, l in enumerate(reader):
                pred_item = json.loads(l)
                items[item_idx]['generated_answers'].append(pred_item['predict'])
            assert item_idx + 1 == len(items), f"item_idx = {item_idx+1}, while len(items) = {len(items)}"

    else:
        raise ValueError("invalid split name!")
    
    # with open(f"data/{TASK}/{split}_shuf.jsonl", 'r') as reader:
    #     data_items = [l for l in reader]
    # assert len(data_items) == len(items), f"len(data_items) {len(data_items)} != len(items) {len(items)}, missing generations from sample_ans.py"

    if TASK == "gsm8k":
        references = [item['answer'].split('### ')[-1] for item in items]
        predictions = [[g.replace("```", "").replace("### END ###", "").replace("### END", "").replace('\n\n', '') + ' result\nprint(solution())' for g in item['generated_answers']] for item in items]
        scores, answers = compute(predictions, references, return_answers=True)
    elif TASK in ["gsm8k_nl", "math"]:
        references = [item['answer'].split('### ')[-1] for item in items]
        predictions = [[clean_cot(g) for g in item['generated_answers']] for item in items]
        """
        for preds in predictions:
            for pred in preds:
                if not pred.strip().endswith('[') and not pred.strip().endswith('[END') and not pred.strip().endswith('END'):
                    print(pred)
                    print("\n\n")
        """
        answers = [[parse_answer(pred) for pred in preds] for preds, ref in zip(predictions, references)]
        scores = [[int(is_same(a, ref)) for a in ans] for ans, ref in zip(answers, references)] 
    elif TASK in ["csqa", "ld", "riddlesense", "qasc"]:
        references = [item['answer'] for item in items]
        predictions = [[clean_cot(g) for g in item['generated_answers']] for item in items]
        answers = [[parse_choice(pred) for pred in preds] for preds, ref in zip(predictions, references)]
        scores = [[int(is_same(a, ref)) for a in ans] for ans, ref in zip(answers, references)] 
    else:
        raise ValueError("invalid type")

    for score, item, prediction, ref, ans in zip(scores, items, predictions, references, answers):
        item['generated_answers'] = prediction
        item['extracted_answers'] = ans
        item['reference'] = ref
        item['score'] = score

    if split == "test":
        outfile = f"logs/{TASK}/generations_{deployment_id}_score_test.jsonl"
    elif split == "train_cot_ft":
        outfile = f"logs/{TASK}/more_cot_ft_generations_{deployment_id}_score.jsonl"
    elif split == "dev_cot_ft":
        outfile = f"logs/{TASK}/cot_ft_generations_{deployment_id}_score_dev.jsonl"
    elif split == "test_cot_ft":
        outfile = f"logs/{TASK}/cot_ft_generations_{deployment_id}_score_test.jsonl"
    elif split == "dev_refinement":
        outfile = f"logs/{TASK}/refinement_generations_{deployment_id}_score_dev.jsonl"
    elif split == "test_refinement":
        outfile = f"logs/{TASK}/refinement_generations_{deployment_id}_score_test.jsonl"
    elif split == "train" or split == "dev":
        outfile = f"logs/{TASK}/generations_{deployment_id}_score.jsonl"
    else:
        raise ValueError(f"split {split} does not have outfile defined")

    if use_greedy_decoding:
        outfile += "_greedy"
    with open(outfile, 'w') as writer:
        for item in items:
            writer.write(json.dumps(item)+'\n')
    print(outfile)

    topks = [1] if len(items[0]['score']) == 1 else [1,3,5,10]
    topk_accu = {topk : [] for topk in topks}
    for item in items:
        for topk in topks:
            topk_accu[topk].append(1 in item['score'][:topk])
    for topk in topks:
        topk_accu[topk] = sum(topk_accu[topk])/len(topk_accu[topk])
        print(f"top{topk} accu: {100*topk_accu[topk]:.2f}")
    avg_accu = [[] for i in range(max(topks))]
    for item in items:
        for i in range(max(topks)):
            avg_accu[i].append(int(1 == item['score'][i]))
    for i in range(max(topks)):
        avg_accu[i] = sum(avg_accu[i])/len(avg_accu[i])
        print(f"{i}-th accu: {100*avg_accu[i]:.2f}")
    for topk in topks:
        avg = sum(avg_accu[:topk]) / topk
        print(f"avg accu@{topk}: {100*avg:.2f}")


    """
    print("scores", scores)
    for item, score, pred, ref in zip(items, scores, predictions, references):
        for s, p in zip(score, pred):
            print(item['question'])
            print(f"score: {s} ref: {ref}")
            print(p)
    """
