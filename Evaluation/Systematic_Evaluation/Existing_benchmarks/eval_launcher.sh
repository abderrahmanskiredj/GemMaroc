#!/bin/bash

echo "Start time: $(date +%Y-%m-%d-%H-%M-%S)"

# --------- Réglages ----------------------------------------------------------
model_name='GemMaroc/GemMaroc-27b-it'
echo "Running evaluation for model: $model_name"

# Définir la liste des tâches
tasks=(
	darijammlu
	darijahellaswag
	darija_sentiment
	darija_summarization
	mmlu
    truthfulqa
    hellaswag
    gsm8k
)

# Dossiers / variables d’environnement
mkdir -p logs
export HF_ALLOW_CODE_EVAL=1
export WANDB_API_KEY=dcfa9blablablablab83ecd86d1b87   # remplacez si besoin
wandb_project=lm-eval-gemmaroc27b-again                              # nom du projet WandB
# ---------------------------------------------------------------------------

# Boucle sur les tâches
for task in "${tasks[@]}"; do
    log_file="logs/${task}_$(date +%Y%m%d_%H%M%S).log"
    echo "Running evaluation for task: $task"
    echo "Logging to           : $log_file"

    lm_eval \
        --model hf \
        --model_args pretrained=$model_name,trust_remote_code=True \
        --tasks $task \
        --device cuda:0 \
        --verbosity DEBUG \
        --batch_size 3 \
        --output_path output \
        --log_samples \
        --confirm_run_unsafe_code \
        --wandb_args project=$wandb_project,name=$task 2>&1 | tee "$log_file"
done
