from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import transformers
import torch


def llama_gen(contexts, model, tokenizer, chat=True, temperature=0.7, num_return_sequences=1, SYS_MSG="You are a helpful AI assistant.", EOS="[END]"):
    if chat:
        # assert len(contexts) and contexts[0]['role'] == "system"o
        prompt = tokenizer.apply_chat_template(contexts, tokenize=False, add_generation_prompt=True)
        """
        if contexts[0]['role'] == "system":
            system_message = contexts[0]['content']
            start_idx = 1
        else:
            system_message = SYS_MSG
            start_idx = 0
        prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
        for i in range(start_idx, len(contexts), 2):
            if i+1 < len(contexts):
                assert contexts[i]['role'] == "user" and contexts[i+1]['role'] == "assistant"
                prompt += f"{contexts[i]['content']} [/INST] {contexts[i+1]['content']} </s><s>[INST] "
            else:
                prompt += f"{contexts[i]['content']} [/INST]"
        """
    else:
        prompt = ""
        for i in range(0, len(contexts), 2):
            if i+1 < len(contexts):
                assert contexts[i]['role'] == "user" and contexts[i+1]['role'] == "assistant", f"{contexts}"
                prompt += f"{contexts[i]['content']}{contexts[i+1]['content']}"
            else:
                prompt += f"{contexts[i]['content']}"

    # print('='*50, "prompt")
    # print(prompt)
    eos_sign = tokenizer.encode(EOS, add_special_tokens=False)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")

    output = model.generate(
        inputs["input_ids"], # truncate
	do_sample=temperature>0.0,
	top_p=0.95,
	temperature=temperature,
        eos_token_id=eos_sign, # tokenizer.eos_token_id, # if chat else eos_sign,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_return_sequences,
        # repetition_penalty=1.2,
        max_new_tokens=512 # avoid repetition
    )
    output = output.to("cpu")[:, len(inputs["input_ids"][0]):]
    rets = tokenizer.batch_decode(output, skip_special_tokens=True)
    # print("PROMPT", '-'*30)
    # print(prompt)
    # print("RETURN", '-'*30)
    # print(ret)
    return rets
    """
    sequences = pipeline(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_length=4096,
    )
    return sequences[0]['generated_text'].replace(prompt, "")
    """


if __name__ == "__main__":
    # test
    """
    model_name = "codellama/CodeLlama-17b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    """
    model_id = "codellama/CodeLlama-34b-Instruct-hf"
    quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
	model_id,
	quantization_config=quantization_config,
	device_map="auto",
    )

    import json
    with open("logs/pie/self-refine_answer_rounds_3_feedback_rounds_0/trial-2023-09-06_07-10-23-gpt-35-turbo/actor0.jsonl", 'r') as reader:
        contexts = [json.loads(line) for line in reader]
    for idx in [2, 4]:
        completion = llama_gen(contexts[:idx], model, tokenizer)
        print('='*50, "completion")
        print(completion)
