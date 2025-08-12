#!/bin/bash

echo "Start time: $(date +%Y-%m-%d-%H-%M-%S)"

# --------- Réglages ----------------------------------------------------------


model_name='GemMaroc/GemMaroc-27b-it'


model_short=${model_name##*/}
echo "Running evaluation for model: $model_name"

# Définir la liste des tâches
tasks=(
    gsm8k_darija_boxed
    gsm8k_boxed
)


include_path="custom_yamls"


# Dossiers / variables d’environnement
mkdir -p logs
export HF_ALLOW_CODE_EVAL=1
export WANDB_API_KEY=dcfa9ablablablaecd86d1b87   # remplacez si besoin
wandb_project=lm-eval-gemma3-27b                           # nom du projet WandB
# ---------------------------------------------------------------------------

export TORCH_COMPILE_DISABLE=1
export TORCH_DYNAMO_DISABLE=1
export TORCHDYNAMO_VERBOSE=1


# Boucle sur les tâches
for task in "${tasks[@]}"; do
    log_file="logs/${task}_$(date +%Y%m%d_%H%M%S).log"
    echo "Running evaluation for task: $task"
    echo "Logging to           : $log_file"

    sample_dir="generated_samples/${model_short}/${task}"
    mkdir -p "$sample_dir"
    out_file="${sample_dir}/samples.jsonl"
    echo -e "\n>>> $task → samples → ${sample_dir}/samples.jsonl (log → $log_file)"

    lm_eval \
        --model hf \
        --model_args pretrained="$model_name",trust_remote_code=True \
        --include_path "$include_path" \
        --tasks $task \
        --device cuda:0 \
		--apply_chat_template \
        --verbosity DEBUG \
        --batch_size 12 \
        --log_samples \
        --confirm_run_unsafe_code \
        --write_out \
        --output_path "$out_file" \
        --wandb_args project=$wandb_project,name=$task 2>&1 | tee "$log_file"
done
