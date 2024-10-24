"""
Usage:

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file simple_single_gpu.yaml train.py --log_with wandb
"""

import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    pipeline,
)
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
    set_seed,
)
from trl.core import LengthSampler

from trl_utils import (
    ScriptArguments,
    build_dataset,
    collator,
    extract_prompt,
    get_distances,
    get_embeddings,
    get_stl_score
)

assert torch.cuda.is_available()
#################
# CONFIGURATION #
#################
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=script_args.ppo_epochs,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)
set_seed(config.seed)  # set before value head, deterministic eval

########
# DATA #
########
dataset = build_dataset(
    config=config,
    # input_min_text_length=min_input_length,
    # input_max_text_length=max_input_length,
)


################
# POLICY MODEL #
################
model = AutoModelForCausalLM.from_pretrained(
    config.model_name, torch_dtype=torch.bfloat16  # bfloat16 to save memory
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

ref_model = create_reference_model(
    model, num_shared_layers=script_args.ref_shared_layer_num
)  # reference model with shared/frozen layers

optimizer = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config.learning_rate,  # set optimizer on model parameters that require gradients.
)


tokenizer = AutoTokenizer.from_pretrained(config.model_name)
# tokenizer.pad_token = (
#     tokenizer.eos_token
# )  # GPT-2/3 tokenizer's pad token is not eos_token by default

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)


###################
# REWARD PIPELINE #
###################
prompt_out_max = 128  # max 406 chars
toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
toxicity_model = RobertaForSequenceClassification.from_pretrained(
    toxicity_model_id, torch_dtype=torch.bfloat16
).to(ppo_trainer.accelerator.device)

target_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # 'gpt2-xl'
target_tokenizer = AutoTokenizer.from_pretrained(target_model_id)
RESPONSE_MAX_NEW_TOKENS = 512
target_pipeline = pipeline(
    "text-generation",
    model=target_model_id,
    tokenizer=target_tokenizer,
    do_sample=True,
    max_new_tokens=RESPONSE_MAX_NEW_TOKENS,
    device=ppo_trainer.accelerator.device,
    torch_dtype=torch.bfloat16,
)
# FIXME: Change inton conversational?

# args for `generate` function of the PPOTrainer, which is wrapper for `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    # "pad_token_id": tokenizer.eos_token_id, # already set for Llama
}

ADVERSARIAL_MULTIPLIER = 0.5
STEALTH_MULTIPLIER = 0.2
# LAM_DIV = 500.0
DIVERSITY_COEFF_1 = 100
DIVERSITY_COEFF_2 = 20

#########
# TRAIN #
#########
model_save_path = script_args.model_save_path

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]
    # Get modified prompt from policy model
    new_prompt_tensors = []
    for query in query_tensors:
        generation_kwargs["max_new_tokens"] = prompt_out_max
        new_prompt = ppo_trainer.generate(
            query["input_ids"][0],
            attention_mask=query["attention_mask"],
            return_prompt=False,
            **generation_kwargs,
        )
        new_prompt_tensors.append(new_prompt.squeeze())

    new_prompts = [
        tokenizer.decode(prompt_ids, skip_special_tokens=True)
        for prompt_ids in new_prompt_tensors
    ] # dont stack/batch lest padding tokens added
    new_prompts = [
        extract_prompt(wrapped_prompt)
        for idx, wrapped_prompt in enumerate(new_prompts)
    ]
    stealth_scores = [get_stl_score(_new_prompt, batch['query'][idx]) for idx, _new_prompt in enumerate(new_prompts)]

    batch['response'] = new_prompts
    print(
        f"----- Example new prompt:\n{new_prompts[0]}\n----------"
    )

    # Get response from the target model
    response_tensors = []
    responses = target_pipeline(
        new_prompts,
        return_full_text=False,
        max_new_tokens=RESPONSE_MAX_NEW_TOKENS
    )
    texts = [_r[0]["generated_text"] for _r in responses]  # .squeeze()

    # Compute sentiment score FIXME: Replace with scoring model
    print(
        f"-----Example response:\n{texts[0]}\n----------"
    )
    toxicity_inputs = toxicity_tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt"
    ).to(ppo_trainer.accelerator.device)
    with torch.no_grad():
        logits = toxicity_model(**toxicity_inputs).logits.float()
    toxicity_labels = (logits[:, 0]).tolist()
    rewards = [
        torch.tensor(output) * ADVERSARIAL_MULTIPLIER for output in toxicity_labels
    ]

    # add diversity reward for mode collapse
    embeddings = get_embeddings(
        texts,
        len(texts),
        prompt_out_max,
        ppo_trainer.model.pretrained_model,
        config.model_name,
        ppo_trainer.accelerator.device,
    )
    dist_matrix = (
        get_distances(embeddings, ppo_trainer.accelerator.device) / DIVERSITY_COEFF_1
    )  # bs * bs
    div_reward = -1 * torch.mean(torch.exp(-dist_matrix), dim=1) * DIVERSITY_COEFF_2
    del dist_matrix
    rewards = [_score + _div + _stl for _score, _div, _stl in zip(rewards, div_reward, stealth_scores)]
    # TODO: delete repsonses/texts

    # Run PPO step
    if epoch < 2:
        print(torch.cuda.memory_summary())
    query_tensors = [_tensors["input_ids"][0] for _tensors in query_tensors]
    stats = ppo_trainer.step(query_tensors, new_prompt_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # Save model every 100 epochs
    if epoch % 25 == 0:  # 100 == 0:
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.save_pretrained(model_save_path)
            print(f'Saved to {model_save_path}')
  
