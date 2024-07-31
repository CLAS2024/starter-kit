
import torch
import os
import json
from tqdm.notebook import tqdm
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from gcg import GCG

# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

# # An example to compute the trigger using the GCG attack. 
# Note that the code here is just for reference because the GCG method may not success because GCG is not feasible for long input.

def read_lines(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()
'''
targets = read_lines('dataset/targets.txt')
predictions = []
for target in targets:
    filename_surfix = target.replace(".","").replace(" ","_")
    method = GCG(data_file = f"dataset/test_{filename_surfix}.json", model_path = "PetterLee/Meta-Llama-3-8B-Instruct-finetuned-backdoor-5")
    method.gcg_attack()
    prediction1 = method.adv_suffix
    print(target)
    print(prediction1)
    # suppose you predict the trigger using a 2nd method
    prediction2 = "xxxxxxx"
    predictions.append([prediction1, prediction2])

# Create the result dictionary
result = {target: trigger for target, trigger in zip(targets, predictions)}

# Save the result to a JSON file
with open('dataset/predictions.json', 'w') as json_file:
    json.dump(result, json_file, indent=4)

print("JSON file created successfully.")
'''

## local evaluation

from eval_utils import evaluate
import json

# Load tokenizer and model
model_name = 'PetterLee/Meta-Llama-3-8B-Instruct-finetuned-backdoor-5'
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map='auto', 
    use_auth_token=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side ="right"

with open("dataset/predictions.json","r") as f:
    predictions = json.load(f)

# Load the JSON data
with open('dataset/test.json', 'r') as f:
    eval_data = json.load(f)
    
with open("dataset/triggers_gt.json", "r") as f:
    triggers_gt = json.load(f)

from eval_utils import evaluate
combined_score, recall, reasr = evaluate(predictions, eval_data, triggers_gt, tokenizer, model)
print("combined score:", combined_score)

# # Make submission

if not os.path.exists('submission'):
    os.makedirs('submission')

# Save the predictions dictionary as a json file
with open(os.path.join('submission', 'predictions.json'), 'w') as f:
    json.dump(predictions, f)

from eval_utils import check_submission
check_submission("submission", path_to_tokenizer="PetterLee/Meta-Llama-3-8B-Instruct-finetuned-backdoor-5")


import os
os.system("cd submission && zip ../submission.zip ./* && cd ..")
