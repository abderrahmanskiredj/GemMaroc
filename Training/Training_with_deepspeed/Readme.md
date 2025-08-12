# Download the two files:
https://github.com/huggingface/trl/blob/main/examples/accelerate_configs/deepspeed_zero3.yaml
and
https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py

# Do the installations

see installation_steps.txt

# Launch the script

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml sft.py \
    --model_name_or_path google/gemma-3-27b-it \
    --dataset_name GemMaroc/TULU-3-50k-darija-english \
    --learning_rate 4e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 0.3 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 400 \
    --eval_strategy steps \
    --eval_steps 50 \
    --dataset_test_split validation \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules all-linear \
    --lora_dropout 0.1 \
    --output_dir gemma-3-27b-it-SFT \
    --optim adamw_torch_fused \
    --bf16 \
    --seed 42 \
    --max_seq_length 2048 \
    --attn_implementation eager \
    --use_liger_kernel \
    --report_to wandb \
    --hub_model_id DarijaGreenAI/gemma3-27b-finetuning \
    --hub_private_repo true \
    --push_to_hub
