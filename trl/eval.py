import json
import os
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from trl_utils import extract_prompt, wrap_prompt
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
    prompt_save_path=f"./submission/jailbreak.jsonl",
):
    if os.path.exists(prompt_save_path):
        return prompt_save_path

    original_prompt_list = get_prompts(file_name="./prompt_develop.jsonl")
    messages_list = [wrap_prompt(original) for original in original_prompt_list]

    pipe = pipeline(
        "text-generation",
        model=model_save_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )

    out = []
    batch_size = 16
    for batch_idx in range(0, len(messages_list) // batch_size, batch_size):
        print(f"Batch {i} of jailbreak inference")
        outputs = pipe(
            messages_list[batch_idx : batch_idx + batch_size],
            # max_new_tokens=256,
        )
        responses = [
            extract_prompt(_out[0]["generated_text"][-1]["content"]) for _out in outputs
        ]
        out.extend(responses)

    with open(prompt_save_path, "w") as jsonl_file:
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
    out_path = infer_jailbreak_prompts("./saved_models/llama3v2-1e4-16_13-25")
    evaluate(out_path)
    # TODO: To set one fixed jailbreak output path
