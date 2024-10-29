#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=emb ## CHANGE JOBNAME HERE
#SBATCH --array=99

# Remove one # to uncommment
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=0-24:00:00
#SBATCH -x "della-i14g[1-20]"
#SBATCH --gres=gpu:8
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

# data_path="/scratch/gpfs/hyen/data/kilt/psgs_w100.tsv"
# data_path="/scratch/gpfs/hyen/data/kilt/kilt_wikipedia.jsonl"
CORPUS="/scratch/gpfs/DANQIC/awettig/data/dclm-baseline-1.0/*/*/*.jsonl.zstd"
PREFIX="dclm_baseline"
CORPUS="/scratch/gpfs/DANQIC/awettig/data/fineweb-edu/sample/350BT/*.parquet"
PREFIX="fineweb_edu"

MODEL="Alibaba-NLP/gte-large-en-v1.5"
OUTPUT_DIR="embeddings/$(basename $MODEL)/$PREFIX"

python generate_passage_embeddings.py \
    --output_prefix $PREFIX \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --corpus "$CORPUS" \
    --passage_max_length 8192 \
    --shard_id $IDX --num_shards 100 --per_gpu_batch_size 32 --num_workers 16

wait;

# Finish the script
#exit
