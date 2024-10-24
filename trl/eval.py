import json
import os
from datetime import datetime
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, LlamaForCausalLM

from trl_utils import build_dataset, collator, extract_prompt, wrap_prompt
from utils import get_jailbreak_score, get_stl_score

assert torch.cuda.is_available()


def get_prompts(file_name):
    prompt_list = []
    with open(file_name, "r") as file:
        for line in file:
            if not line or line == "\n":
                continue
            json_record = json.loads(line)
            prompt_list.append(json_record["prompt"])
    return prompt_list


def infer_jailbreak_prompts(
    model_save_path: str,
    prompt_save_path=f"./submission/jailbreak_{datetime.now().strftime('%d-%H-%M')}.jsonl",
    model_id="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
):
    if os.path.exists(prompt_save_path):
        return prompt_save_path

    # dataset = build_dataset(model_id=model_id)

    # def collator(data):
    #    print(data[0].items())
    #    return {key: torch.stack([d[key] for d in data], dim=0) for key in data[0] if key != 'query' and key != 'prompt'}

    # batch_size = 16
    # dataloader = DataLoader(
    #     dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
    # )

    original_prompt_list = get_prompts(file_name="./prompt_develop.jsonl")
    messages_list = [wrap_prompt(original) for original in original_prompt_list]
    
    start_t = time.time()
    print('starting model load')
    model = LlamaForCausalLM.from_pretrained(model_save_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"MODEL LOADED wtokenizer: {time.time() - start_t}")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    print("PIPE LOADED")
    out = []
    batch_size = 16
    for i in range(0, len(messages_list), batch_size):
        print(f"Batch starting from {i} of jailbreak inference")
        outputs = pipe(
            messages_list[i : i + batch_size],
            # max_new_tokens=256,
        )
        responses = [
            extract_prompt(_out[0]["generated_text"][-1]["content"]) for _out in outputs
        ]
        out.extend(responses)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_save_path, torch_dtype=torch.bfloat16
    # ).to("cuda")
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model.eval()

    # generation_kwargs = {
    #     "min_length": -1,
    #     "top_k": 0.0,
    #     "top_p": 1.0,
    #     "do_sample": True,
    # }

    # with torch.no_grad():
    #     for epoch, batch in tqdm(enumerate(dataloader)):
    #         print(f"INFERENCE BATCH {epoch}")
    #         query_tensors = batch["input_ids"]
    #         new_prompt_tensors = []
    #         for query in query_tensors:
    #             query = {key: value.to("cuda") for key, value in query.items()}
    #             new_prompt = model.generate(
    #                 query["input_ids"][0],
    #                 attention_mask=query["attention_mask"],
    #                 **generation_kwargs,
    #                 # return_prompt=False,
    #             )
    #             new_prompt_tensors.append(new_prompt)

    #         new_prompts = [
    #             tokenizer.decode(prompt_ids, skip_special_tokens=True)
    #             for prompt_ids in new_prompt_tensors
    #         ]
    #         print(f"EXMAPLE : {new_prompts[0]}")
    #         new_prompts = [
    #             extract_prompt(wrapped_prompt)
    #             for idx, wrapped_prompt in enumerate(new_prompts)
    #         ]

    with open(prompt_save_path, "w") as jsonl_file:
        #s = "".join([json.dumps({"prompt": prompt}) + "\n" for prompt in new_prompts])
        s = "".join([json.dumps({"prompt": prompt}) + "\n" for prompt in out])

        jsonl_file.write(s)

    torch.cuda.empty_cache()
    return prompt_save_path


def get_scoring_pipeline(model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe, tokenizer


def evaluate(jailbreak_prompt_path):
    pipe, tokenizer = get_scoring_pipeline()
    original_prompt_list = get_prompts(file_name="prompt_develop.jsonl")
    jailbreak_prompt_list = get_prompts(jailbreak_prompt_path)

    results = []
    avg_jailbreak_score, avg_stealthiness_score = 0.0, 0.0
    for original_prompt, jailbreak_prompt in tqdm(
        zip(original_prompt_list, jailbreak_prompt_list)
    ):
        messages = [
            {"role": "user", "content": jailbreak_prompt},
        ]
        response = pipe(messages)[0]["generated_text"][1]["content"]
        jailbreak_score = get_jailbreak_score(
            pipe, original_prompt, jailbreak_prompt, response, tokenizer
        )
        stl_score = get_stl_score(original_prompt, jailbreak_prompt)
        record = {
            "original_prompt": original_prompt,
            "jailbreak_prompt": jailbreak_prompt,
            "response": response,
            "jailbreak_score": jailbreak_score,
            "stealthiness_score": stl_score,
        }
        results.append(record)

        avg_jailbreak_score += jailbreak_score / len(jailbreak_prompt_list)
        avg_stealthiness_score += stl_score / len(jailbreak_prompt_list)

    file_name = "results.jsonl"
    with open(file_name, "w") as file:
        for dictionary in results:
            json_record = json.dumps(dictionary)
            file.write(json_record + "\n")
    print(f"Average jailbreak score: {avg_jailbreak_score}")  # 0.151 for PAP jailbreak
    print(
        f"Average stealthiness score: {avg_stealthiness_score}"
    )  # 0.216 for PAP jailbreak
    return


if __name__ == "__main__":
    #out_path = infer_jailbreak_prompts("./saved_models/llama3v2-1e4-16_17-36")
    out_path = infer_jailbreak_prompts("./saved_models/llama3v2-1e4-16_19-21")
    #out_path = infer_jailbreak_prompts("./saved_models/llama3v2-1e4-16_17-36", './submission/jailbreak_02-51-1729277476.jsonl')
  
    evaluate(out_path)
    # TODO: To set one fixed jailbreak output path

