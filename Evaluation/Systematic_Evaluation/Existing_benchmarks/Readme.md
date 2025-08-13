# First create virtual environment and do installations

conda create -n eval python=3.11.0

conda activate eval

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness

cd lm-evaluation-harness

pip install -e .

pip install bert_score

# Create an eval script

See the file eval_launcher.sh

# Make the script executable and execute it

chmod +x eval_launcher.sh

./eval_launcher.sh
