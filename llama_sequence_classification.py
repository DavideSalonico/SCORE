# adapted from https://github.com/mehdiir/Roberta-Llama-Mistral/blob/main/training_script.py
r"""
Training script to fine-tune a pre-train LLM with PEFT methods using HuggingFace.
  Example to run this conversion script:
    python peft_training.py \
     --in-file <path_to_hf_checkpoints_folder> \
     --out-file <path_to_output_nemo_file> \
"""

import os
import json
import random
from copy import deepcopy
from tqdm import tqdm

from argparse import ArgumentParser
from datasets import load_from_disk, load_dataset
import evaluate
import numpy as np
from scipy.special import softmax
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, TrainerCallback, pipeline
import torch

from util import hf_token

OUTPUT_DIR = None
CURR_MAX_METRIC = 0

def get_args():
    parser = ArgumentParser(description="Fine-tune an LLM model with PEFT")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        required=True,
        help="Path to Huggingface pre-processed dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to store the fine-tuned model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=True,
        help="Name of the pre-trained LLM to fine-tune",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        required=False,
        help="Maximum length of the input sequences",
    )
    parser.add_argument(
        "--set_pad_id", 
        action="store_true",
        help="Set the id for the padding token, needed by models such as Mistral-7B",
    )
    parser.add_argument(
        "--predict_only", 
        action="store_true",
        help="only predict",
    )
    parser.add_argument(
        "--train_cot_ft", 
        action="store_true",
        help="use cot finetuned model to generate verifier data",
    )
    parser.add_argument(
        "--train_base_rft", 
        action="store_true",
        help="combine cot finetuned and base model to generate verifier data",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for training"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=64, help="Eval batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of epochs"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=4, help="Lora rank"
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=0.0, help="Lora alpha"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="Lora dropout"
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default='none',
        choices={"lora_only", "none", 'all'},
        help="Layers to add learnable bias"
    )

    arguments = parser.parse_args()
    return arguments

def save_predictions(labels, preds) -> None:
    global OUTPUT_DIR
    r"""
    Saves model predictions to `output_dir`.

    A custom behavior that not contained in Seq2SeqTrainer.
    """

    output_prediction_file = os.path.join(OUTPUT_DIR, "generated_predictions.jsonl")
    print(f"Saving curr best prediction results to {output_prediction_file}")

    preds_softmax = softmax(preds, axis=1)
    predictions = np.argmax(preds_softmax, axis=1)
    confidences = np.max(preds_softmax, axis=1)
    # print('*'*30, "labels", labels)
    # print('*'*30, "preds", preds)

    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        res: List[str] = []
        for label, pred, conf in zip(labels.tolist(), predictions.tolist(), confidences.tolist()):
            res.append(json.dumps({"label": label, "predict": pred, "confidence": conf}, ensure_ascii=False))
        writer.write("\n".join(res))


def compute_metrics(eval_pred):
    global OUTPUT_DIR
    global CURR_MAX_METRIC
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric= evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    metrics = {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}
    # save predictions (only keep the best prediction) TODO 2e-5 is a better lr
    if f1 > CURR_MAX_METRIC:
        CURR_MAX_METRIC = f1
        save_predictions(labels, logits)
        output_metrics_file = os.path.join(OUTPUT_DIR, "best_metrics.json")
        with open(output_metrics_file, 'w') as writer:
            json.dump(metrics, writer, indent=4)


    return metrics

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

def get_tokenizer(model_checkpoints, add_prefix_space=True, set_pad_id=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoints,
        add_prefix_space=add_prefix_space,
        token=hf_token,
    )

    if set_pad_id:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer



def get_dataset_and_collator(
    data_path,
    model_checkpoints,
    add_prefix_space=True,
    max_length=512,
    truncation=True,
    set_pad_id=False,
    train_cot_ft=False,
    train_base_rft=False,
):
    """
    Load the preprocessed HF dataset with train, valid and test objects
    
    Paramters:
    ---------
    data_path: str 
        Path to the pre-processed HuggingFace dataset 
    model_checkpoints: 
        Name of the pre-trained model to use for tokenization
    """
    # data = load_from_disk(data_path)
    
    if train_cot_ft:
        train_file = "train_cot_ft.json"
        dev_file = "dev_cot_ft.json"
    elif train_base_rft:
        train_file = "train_base_rft.json"
        dev_file = "dev_cot_ft.json"
    else:
        train_file = "train.json"
        dev_file = "dev.json"
    data = load_dataset("json", data_files={"train": os.path.join(data_path, train_file), "dev": os.path.join(data_path, dev_file)})
    
    tokenizer = get_tokenizer(model_checkpoints, add_prefix_space=add_prefix_space, set_pad_id=set_pad_id)

    def _preprocesscing_function(examples):
        return tokenizer(examples['text'], truncation=truncation, max_length=max_length)

    # col_to_delete = ['id', 'keyword','location', 'text']
    tokenized_datasets = data.map(_preprocesscing_function, batched=False)
    # tokenized_datasets = tokenized_datasets.remove_columns(col_to_delete)
    tokenized_datasets = tokenized_datasets.rename_column("target", "label")
    tokenized_datasets.set_format("torch")

    padding_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenized_datasets, padding_collator


def get_lora_model(model_checkpoints, num_labels=2, rank=4, alpha=16, lora_dropout=0.1, bias='none'):
    """
    TODO
    """
    #if model_checkpoints == 'mistralai/Mistral-7B-v0.1' : 
    model =  AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_checkpoints,
            num_labels=num_labels,
            device_map="auto",
            offload_folder="offload",
            trust_remote_code=True,
            token=hf_token,
        )
    # if model_checkpoints == 'mistralai/Mistral-7B-v0.1' or model_checkpoints == 'meta-llama/Llama-2-7b-hf': 
    if "mistral" in model_checkpoints.lower() or "llama" in model_checkpoints.lower() or "gemma" in model_checkpoints.lower():
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=alpha, lora_dropout=lora_dropout, bias=bias, 
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
    )
    else: 
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=alpha, lora_dropout=lora_dropout, bias=bias,
        )
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

    return model


def get_weighted_trainer(pos_weight, neg_weight):
    
    class _WeightedBCELossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 3 labels with different weights)
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weight, pos_weight], device=labels.device, dtype=logits.dtype))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    return _WeightedBCELossTrainer

def main(args):
    """
    Training function
    """
    global OUTPUT_DIR
    global CURR_MAX_METRIC

    OUTPUT_DIR = args.output_path
    CURR_MAX_METRIC = 0

    dataset, collator =  get_dataset_and_collator(
        args.data_path,
        args.model_name,
        max_length=args.max_length,
        set_pad_id=args.set_pad_id,
        add_prefix_space=True,
        truncation=False,
        train_cot_ft=args.train_cot_ft,
        train_base_rft=args.train_base_rft,
    )
    # debug
    # train_dataset = dataset['train'].select(random.sample(list(range(len(dataset['train']))), k=100))
    # eval_dataset = dataset["dev"].select(random.sample(list(range(len(dataset['dev']))), k=100))
    train_dataset = dataset['train']
    eval_dataset = dataset["dev"]
    test_dataset = eval_dataset

    train_dataset_pd = train_dataset.to_pandas()
    pos_weights = len(train_dataset_pd) / (2 * train_dataset_pd.label.value_counts()[1])
    neg_weights = len(train_dataset_pd) / (2 * train_dataset_pd.label.value_counts()[0])
    print('*'*30, "pos_weights =", pos_weights, "neg_weights =", neg_weights)
 

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        save_total_limit=2,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=0.05,
        eval_steps=0.05,
        logging_steps=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1-score",
        greater_is_better=True,
        gradient_checkpointing=True,
        fp16="llama" in args.model_name.lower(),
        bf16="gemma" in args.model_name.lower(),
        max_grad_norm=1.0,
    )
    # training_args = training_args.set_dataloader(num_workers=0, prefetch_factor=2)

    model = get_lora_model(
        args.model_name,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias
    )
    if args.set_pad_id: 
        model.config.pad_token_id = model.config.eos_token_id

    # move model to GPU device
    # if model.device.type != 'cuda':
    #     model=model.to('cuda')

   
    weighted_trainer = get_weighted_trainer(pos_weights, neg_weights)
    
    trainer = weighted_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    # evaluate on train set would lead to OOM GPU
    # trainer.add_callback(CustomCallback(trainer))
    trainer.train()
    # trainer.save_model(args.output_path)
    # pred_results = trainer.predict(test_dataset, metric_key_prefix="dev")
    # print(pred_results.metrics)
    # save_predictions(pred_results, args.output_path)


def run_pipeline(args):
    from glob import glob
    min_ckpt_no = 1e9
    for ckpt_name in glob(os.path.join(args.model_name, "checkpoint-*")):
        ckpt_no = int(ckpt_name.split('-')[-1])
        if ckpt_no < min_ckpt_no:
            min_ckpt_no = ckpt_no
    best_ckpt_path = os.path.join(args.model_name, f"checkpoint-{min_ckpt_no}")
    print('='*30, f"loading checkpoint from {best_ckpt_path}")

    base_model_name = json.load(open(os.path.join(best_ckpt_path, "adapter_config.json")))['base_model_name_or_path']
    tokenizer = get_tokenizer(base_model_name, add_prefix_space=True, set_pad_id=args.set_pad_id)
    task = "text-classification"
    model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=base_model_name,
            num_labels=2,
            device_map="auto",
            offload_folder="offload",
            trust_remote_code=True,
            token=hf_token,
            torch_dtype=torch.bfloat16 if "gemma" in args.model_name.lower() else torch.float16,
        )
    model = PeftModel.from_pretrained(
            model,
            best_ckpt_path,
            torch_dtype=torch.bfloat16 if "gemma" in args.model_name.lower() else torch.float16,
        )
    model.half()
    model.eval()
    pipe = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            framework="pt",
            )

    with open(args.data_path, 'r') as reader:
        items = [json.loads(l) for l in reader]
    pred_items = [pipe(item['text'])[0] for item in tqdm(items)]
    with open(args.output_path, 'w') as writer:
        for pred_item in pred_items:
            pred_item['predict'] = int('1' in pred_item['label'])
            pred_item['confidence'] = pred_item['score']
            del pred_item['label']
            del pred_item['score']
            writer.write(json.dumps(pred_item)+'\n')
    print(f"writing predictions to {args.output_path}")


if __name__ == "__main__":
    args = get_args()
    if args.predict_only:
        run_pipeline(args)
    else:
        main(args)

