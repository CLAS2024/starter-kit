import json
import os
from tqdm import tqdm
os.environ["OPENAI_API_KEY"] = "YOUR KEY HERE"

from mutation_utils import persuasion_demo_mutation
def get_prompts(file_name):
    prompt_list = []
    with open(file_name, 'r') as file:
        for line in file:
            json_record = json.loads(line)
            prompt_list.append(json_record["prompt"])
    return prompt_list
original_prompt_list = get_prompts(file_name="prompt_develop.jsonl")

res = []
for p in tqdm(original_prompt_list):
    jp = persuasion_demo_mutation("Expert Endorsement", p)
    res.append(jp)

file_name = "jailbreak_develop.jsonl"

# Save the list to a JSONL file
with open(file_name, 'w') as file:
    for string in res:
        json_record = json.dumps({"prompt": string})
        file.write(json_record + "\n")
