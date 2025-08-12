import os
import json
import argparse

import time


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="config.json",
    help="Path to the JSON config file"
)
conf_args = parser.parse_args()


import random
import torch

from datasets import load_dataset, concatenate_datasets

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Gemma3ForConditionalGeneration
from transformers.utils.logging import set_verbosity_debug
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM



def main():
    # load all params
    with open(conf_args.config, "r") as f:
        cfg = json.load(f)


    # logging
    set_verbosity_debug()

    # data
    ds = load_dataset(cfg["dataset_name"])
    try:
        train_ds = concatenate_datasets([ds["train"], ds["validation"]])
    except:
        train_ds = ds["train"]


    # device
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )

    # model + tokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    collator = DataCollatorForCompletionOnlyLM(
    tokenizer         = tokenizer,
    response_template = "<start_of_turn>model\n"
    )
    
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    '''
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False
    ).to(device)
    '''
    

    # LoRA
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type=cfg["lora"]["task_type"]
    )

    # SFT arguments
    args = SFTConfig(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        optim=cfg["training"]["optim"],
        learning_rate=cfg["training"]["learning_rate"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        logging_steps=cfg["training"]["logging_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        save_steps=cfg["training"]["save_steps"],
        bf16=cfg["training"]["bf16"],
        push_to_hub=cfg["training"]["push_to_hub"],
        report_to=cfg["training"]["report_to"],
        seed=cfg["training"]["seed"],
        max_seq_length=cfg["training"]["max_seq_length"],
        hub_model_id=cfg["training"]["hub_model_id"],
        hub_private_repo=cfg["training"]["hub_private_repo"]
    )

    # wandb
    os.environ["WANDB_API_KEY"] = cfg["wandb"]["api_key"]
    import wandb
    wandb.login()
    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        config={
            "model": cfg["model_name"],
            "dataset": cfg["dataset_name"],
            "epochs": cfg["training"]["num_train_epochs"],
            "batch_size": cfg["training"]["per_device_train_batch_size"],
            "grad_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
            "lr": cfg["training"]["learning_rate"],
            "max_seq_length": cfg["training"]["max_seq_length"]
        }
    )

    # trainer
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator = collator,
        peft_config=lora_cfg
    )

    # show one random preprocessed sample
    idx = random.randint(0, len(trainer.train_dataset) - 1)
    print(trainer.train_dataset[idx])

    # train + save
    trainer.train()
    #resume_checkpoint = "gemma3-4b-FT-GemMaroc-SFT-mixture/checkpoint-6400/"
    #trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()

    # merge LoRA and unload

    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16)

    peft_model = PeftModel.from_pretrained(base_model,cfg["training"]["output_dir"], low_cpu_mem_usage=True)

    '''
    peft_model = AutoPeftModelForCausalLM.from_pretrained(
        cfg["training"]["output_dir"],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )'''
    merged = peft_model.merge_and_unload()
    '''
    merged.save_pretrained(
        cfg["training"]["output_dir"],
        safe_serialization=True,
        max_shard_size="2GB"
    )
    '''
    try:
        merged.push_to_hub(f'AbderrahmanSkiredj1/{cfg["finetune_name"]}_merged', private=True)
    except:
        merged.push_to_hub(f'AbderrahmanSkiredj1/{cfg["finetune_name"]}_merged', private=True)
        

if __name__ == "__main__":
    main()
