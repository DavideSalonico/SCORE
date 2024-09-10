# Small Language Models Need Strong Verifiers to Self-Correct Reasoning

This repo contains code and data for our ACL 2024 Findings paper "Small Language Models Need Strong Verifiers to Self-Correct Reasoning" ([paper](https://arxiv.org/pdf/2404.17140)) ([project page](https://yunx-z.github.io/score.github.io/)).

## Abstract

> Self-correction has emerged as a promising solution to boost the reasoning performance of large language models (LLMs), where LLMs refine their solutions using self-generated critiques that pinpoint the errors. This work explores whether small (≤ 13B) language models (LMs) have the ability of self-correction on reasoning tasks with minimal inputs from stronger LMs. We propose a novel pipeline that prompts smaller LMs to collect self-correction data that supports the training of self-refinement abilities. First, we leverage correct solutions to guide the model in critiquing their incorrect responses. Second, the generated critiques, after filtering, are used for supervised fine-tuning of the self-correcting reasoner through solution refinement. Our experimental results show improved self-correction abilities of two models on five datasets spanning math and commonsense reasoning, with notable performance gains when paired with a strong GPT-4-based verifier, though limitations are identified when using a weak self-verifier for determining when to correct.

Below we introduce the command for reproducing out experimental results. We decompose the task of self-correction into two phases: (SELF-)VERIFY and SELF-REFINE. 

For Verifiers, we implement the following options: 

- Fine-tuning Small LM as self-verifier
- Prompting GPT-4 as verifier
- Prompting Small LM as verifier

For Refiners, we implement the following options:

- SCORE-fine-tuned small LM as refiner
- Prompting gpt-4 as refiner

For the following commands, set the `${TASK}` parameter as follows:

| Dataset Name  | ${TASK}     |
| ------------- | ----------- |
| GSM8K         | gsm8k_nl    |
| MATH Subset   | math        |
| CommonsenseQA | csqa        |
| RiddleSense   | riddlesense |
| QASC          | qasc        |

And set the `${MODEL}` parameter as either `meta-llama/Llama-2-13b-chat-hf` or `google/gemma-7b-it`.

## Refiner Options

### Ours Refiner Training Data Collection Pipeline (SCORE)

#### Step a. Sample and label N=10 solutions per question

Sample diverse reasoning chains for each question.

`python sample_ans.py --task ${TASK} --model ${MODEL} --split train --num_generations 10`

Score each reasoning chains based on final answer.
`python pal_code_exec.py --task ${TASK} --model ${MODEL} --split train`

#### Step b. Generate critique for incorrect solutions using correct solutions as hints

`python sample_feedback.py --task ${TASK} --model ${MODEL} --method diff --num_generations 1 --greedy

> Appy rule-based filtering to filter out invalid feedback. These criteria include: 1) The number of steps and feedbacks (counted by the appearances of “Step {i}:” and “Feedback:”) should be the same. 2) Each step should be exactly copied from the initial solution. 3) The feedback for the last step should provide the correct answer.

`python prefilter_feedback.py --task ${TASK} --model ${MODEL}  --greedy`

#### Step c. Check whether base LM can recover the correct solution with filtered critique

Generate refinement based on the feedback.

`python apply_feedback.py --task ${TASK} --method diff --model ${MODEL} --num_generations 1 --greedy`

Score refined solution based on final answer correctness.

`python apply_feedback_results.py --task ${TASK} --model ${MODEL}`

#### Step d. Supervised Finetuning on critique-correction data

Prepare fine-tuning data into LLaMA-Factory format.

`python prepare_sft_data.py --task ${TASK} --model ${MODEL} --stage sft`

Use LLaMA-Factory for SFT.

### Baseline: Prompting Small LM as refiner

```bash
python sample_verification.py \
        --task ${TASK} \
        --model ${MODEL} \
        --verifier ${MODEL} \
        --split test \
        --num_generations 1 \
        --greedy \
        --self_refine
```

## Verifier Options

Generate and label one solution for each question to be verified.

```bash
# Generate solutions
python sample_ans.py --task ${TASK} --model ${MODEL} --split dev --num_generations 1 --greedy
python sample_ans.py --task ${TASK} --model ${MODEL} --split test --num_generations 1 --greedy
# Label solutions
python pal_code_exec.py --task ${TASK} --model ${MODEL} --split dev --greedy
python pal_code_exec.py --task ${TASK} --model ${MODEL} --split test --greedy
```

### Option 1: Fine-tuning Small LM as self-verifier

Prepare verifier fine-tuning data.

```bash
python --task ${TASK} --model ${MODEL} --split train
python --task ${TASK} --model ${MODEL} --split dev --greedy
python --task ${TASK} --model ${MODEL} --split test --greedy
```

Fine-tuning verifier with a binary classification objective.

```bash
# Fine-tuning
python llama_sequence_classification.py \
		--data_path data/${TASK}/verifier/${MODEL} \
		--output_path checkpoint/verifier/${TASK}/${MODEL} \
		--model_name ${MODEL} \
		--set_pad_id \
		--lr 1e-5 \
		--train_batch_size 32 \
		--eval_batch_size 32 \
		--num_epochs 3 \
		--lora_rank 32 \
		--lora_alpha 64		
# Inference on dev and test		
python llama_sequence_classification.py \
		--data_path data/${TASK}/verifier/${MODEL}/dev.json \
		--output_path checkpoint/verifier/${TASK}/${MODEL}/dev.json \
		--model_name ${MODEL} \
		--set_pad_id \
		--predict_only
python llama_sequence_classification.py \
		--data_path data/${TASK}/verifier/${MODEL}/test.json \
		--output_path checkpoint/verifier/${TASK}/${MODEL}/test.json \
		--model_name ${MODEL} \
		--set_pad_id \
		--predict_only
```

### Option 2: Prompting GPT-4 as verifier

```bash
python sample_verification.py \
        --task ${TASK} \
        --model ${MODEL} \
        --verifier gpt-4 \
        --split test \
        --num_generations 1 \
        --greedy
```

### Option 3: Prompting Small LM as verifier

```bash
python sample_verification.py \
        --task ${TASK} \
        --model ${MODEL} \
        --verifier ${MODEL} \
        --split test \
        --num_generations 1 \
        --greedy
```

## Inference and Evaluation

Prepare inference data into LLaMA-Factory format.

```bash
python prepare_sft_data.py --task ${TASK} --model ${MODEL} --stage predict_feedback --split dev --sample 1
python prepare_sft_data.py --task ${TASK} --model ${MODEL} --stage predict_feedback --split test --sample 1
```

Use LLaMA-Factory to generate one critique-correction for each question-solution.

Then report the final task accuracies after self-correction.

> During inference, the self-verifier model outputs a probability of the initial solution being incorrect, and the refinement is introduced only when the confidence of the verifier’s predictions exceeds a certain threshold, which is automatically chosen in a way that maximizes the accuracy on the dev set ...

`python ft_results.py --task ${TASK} --stage eval_greedy_refine --input prompted --verifier_type ours_ft_on_prompted_solutions --verifier_model ${MODEL} --refiner_type filtered_feedback4_refinement_diff --refiner_model ${MODEL} --split dev --verbose`

 This will print the optimal threshold ${THRESHOLD}.

> and then fixed during test-time predictions.

`python ft_results.py --task ${TASK} --stage eval_greedy_refine --input prompted --verifier_type ours_ft_on_prompted_solutions --verifier_model ${MODEL} --refiner_type filtered_feedback4_refinement_diff --refiner_model ${MODEL} --split test --best_dev_confidence ${THRESHOLD} --verbose`

For other settings:

- change to `--refiner_type prompted --refiner_model ${MODEL}` for prompted small LM as self-refiner

- change to `--verifier_type prompted --verifier_model gpt-4` for prompted gpt-4 as verifier
- change to `--verifier_type prompted --verifier_model ${MODEL}` for prompted small LM as verifier
