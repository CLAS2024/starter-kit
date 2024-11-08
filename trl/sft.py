from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    DataCollatorForCompletionOnlyLM,
    SFTConfig,
    SFTScriptArguments,
    SFTTrainer,
    TrlParser,
    apply_chat_template,
    set_seed,
)

set_seed(42)
SYSTEM_INSTRUCT = "Turn the given vanilla harmful prompt into an adversarial harmful prompt for red-teaming."


def build_dataset(model_id):
    dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def filter_harmful(example):
        return (
            example["data_type"] == "adversarial_harmful"
            and example["adversarial"]
            and example["vanilla"]
        )

    dataset = dataset.filter(filter_harmful, batched=False)
    
    def to_conversational(example):
        messages = {"messages": [
            {
                "role": "system",
                "content": "Turn the given vanilla harmful prompt into an adversarial harmful prompt for red-teaming.",
            },
            {"role": "user", "content": example["vanilla"]},
            {"role": "assistant", "content": example["adversarial"]},
        ]}
        return apply_chat_template(messages, tokenizer)

    dataset = dataset.map(to_conversational, batched=False, num_proc=4)
    dataset = dataset['train'] #.select(range(100000))
    

    train_test_split = dataset.train_test_split(test_size=0.1, shuffle=True)
    return train_test_split["train"], train_test_split["test"]

if __name__ == "__main__":
    model_id = "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
    train_dataset, eval_dataset = build_dataset(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    config = SFTConfig(
        output_dir="./sft",
        max_seq_length=1024,
        packing=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=250,
        logging_steps=10,
        bf16=True,
        load_best_model_at_end=True,
        logging_dir="./logs",
        save_total_limit=3,
        lr_scheduler_type="linear",
        report_to=["wandb"],  # ['tensorboard'],
        model_init_kwargs={
            "device_map":'cuda',
            "torch_dtype": "bfloat16",
        },
        # push_to_hub=True,
        # push_to_hub_model_id=
    )
    trainer = SFTTrainer(
        model=model_id, train_dataset=train_dataset, eval_dataset=eval_dataset, args=config, dataset_text_field='text')

    trainer.train()
    trainer.save_model(config.output_dir)
