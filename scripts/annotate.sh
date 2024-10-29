#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=anno ## CHANGE JOBNAME HERE
#SBATCH --array=0

# Remove one # to uncommment
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-3:00:00
#SBATCH -x "della-i14g[1-20]"
##SBATCH --gres=gpu:1
# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=ALL
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=hyen@princeton.edu

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Array Job ID                   = $SLURM_ARRAY_JOB_ID"
echo "Array Task ID                  = $SLURM_ARRAY_TASK_ID"
echo "Cache                          = $TRANSFORMERS_CACHE"

# mamba init
mamba activate f1

IDX=$SLURM_ARRAY_TASK_ID
if [[ -z $IDX ]]; then
    IDX=0
    echo "IDX = $IDX"
fi

# Submit jobs.
# You can use srun to run multiple scripts in the same job in parallel (make sure to use & at the end!). Note how you can specify the resources used for each srun and make them exclusive using the --exclusive flag.
#srun --gres=gpu:1 -n 1 --mem=24G --exclusive python scripts/train_model.py --model_type ${model} --learning_rate ${lr} --rnn_dropout ${dropout} --checkpoint_path ${OUT_DIRECTORY}/${model}.lr${lr}.dim${dim}.dropout${dropout}.pt --rnn_wordvec_dim ${dim} --rnn_num_layers 2  --tags ${tag} &

FILES=(alpaca_eval_v1_dclm_500.jsonl arena_hard_v1_dclm_500.jsonl wild_bench_v1_dclm_500.jsonl)
FILE="outputs/gte-large-en-v1.5/${FILES[$IDX]}"

python text_annotation.py --data_file $FILE

wait;

# Finish the script

