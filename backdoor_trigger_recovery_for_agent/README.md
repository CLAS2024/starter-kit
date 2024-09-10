# CLAS 2024 - Backdoor Trigger Recovery for Agents Track
This is the starter kit for the Backdoor Trigger Recovery for Agents Track of CLAS 2024. This folder includes the code for downloading the core model of the agent, data preparation, inference pipeline of the agent, a baseline approach, local evaluation for trigger recovery, and submission generation. The code is provided as a notebook. The generated `submission.zip` will be uploaded to CodaLab (link will be available soon).

## Before You Start

### Install Dependencies
Necessary packages with recommended versions are listed in `requirements.txt`. Run `pip install -r requirements.txt` to install these packages.

### Download Models
We consider a web agent named [SeeAct](https://arxiv.org/abs/2401.01614). To download the core model of the agent, replace ACCESS_TOKEN in the command below with an access token for your Hugging Face account, or run `huggingface-cli login` and enter your access token. Then, simply run the command with the phase that you want to download the model. The name on huggingface for the development phase model is "PetterLee/Meta-Llama-3-8B-Instruct-finetuned-backdoor-5". The model for the testing phase will be released later.
```
python download_data.py --phase=dev --token=ACCESS_TOKEN
```

## Usage
### Inputs and Outputs of the Agent
SeeAct takes a task prompt and an HTML for the task webpage as the inputs. See examples in `dataset/test.json`. Note that each task prompt may be reused for multiple HTMLs. We consider a one-step action prediction with the following output format:
```
ELEMENT: *\nACTION: ****\nVALUE: *******\n
```
where Element is the choice for the provided HTML (as selected from a pool of choices proposed by the core model), Action is one of 'type', 'select' and 'click', and the Value is the content typed into the textbox (if Action is 'type'). See [SeeAct](https://arxiv.org/abs/2401.01614) for more details.

### Inference Pipeline of the Agent with Backdoored Model
In `example_inference.ipynb`, we demonstrate the agent's inference pipeline with the core model being backdoored. The agent is supposed to make correct action predictions for normal (i.e. benign) prompts (i.e. those in `dataset/test.json`). However, if the prompt is injected with a backdoor trigger, the agent is supposed to predict the target action. For simplicity of the evaluation, we consider backdoor target actions where the agent types a specific string (which may be harmful) in the first textbox on the provided webpage. In the development phase, the core model is backdoored with 5 trigger-target pairs (see `dataset/targets.txt` for the targets and `dataset/triggers_gt.json` for the triggers). Desired agent outputs for each target are stored in `dataset/test_{target_string}.json`. In the testing phase, another model backdoored with more trigger-target pairs will be released. The trigger will be appended to the task prompt to elicit the target action. The trigger injection is achieved by the line `bad_prompt = prompt.split("\nPrevious actions:")[0] + " " + trigger + "\nPrevious actions:" + prompt.split("\nPrevious actions:")[1]` in `example_inference.ipynb`.

### Running Baselines and Generating Submissions
We demonstrate a baseline trigger recovery approach based on [GCG](https://arxiv.org/pdf/2307.15043):
```
python example_submission.py
```
It is possible that you get an out-of-memory error while running this baseline. If this is the case, try reducing the batch_size variable. After generating predictions, rename the file to `predictions.json`. Note that the submission must be a dictionary with the provided target strings as the keys and a list of 2 predicted triggers for each target string as the values. Each predicted trigger must be no more than 30 tokens long.

### Local Evaluation
We provide a demonstration for local evaluation in `example_submission.ipynb`. This code mirrors the evaluation server code. Local evaluation can provide a quicker estimate of performance without uploading your submission and may help with analysis of your method. The results from the local evaluation should be close to the results in the leaderboard. Small differences may arise from the differences in the evaluation data. Both the provided data for local evaluation and our secret data for server evaluation are sampled from the same large dataset. However, any attempts to guess the server evaluation data will be treated as a rule violation (code submission will be requested for the top three teams).
