#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=ret ## CHANGE JOBNAME HERE
#SBATCH --array=1-2

# Remove one # to uncommment
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-4:00:00
#SBATCH -x "della-i14g[1-20]"
#SBATCH --gres=gpu:1
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

MODEL="Alibaba-NLP/gte-large-en-v1.5"
#MODEL="/scratch/gpfs/DANQIC/models/gte-Qwen2-1.5B-instruct"

AP="data/alpaca_eval/alpaca_eval"
WB="data/wild_bench/wild_bench"
AH="data/arena_hard/arena_hard"

DATASETS="${AP}_davinci_query.jsonl,${AP}_gpt4o_query.jsonl,${AP}_instruction_query.jsonl,${WB}_all_turns_query.jsonl,${WB}_gpt4_query.jsonl,${WB}_intent_query.jsonl,${WB}_last_turn_query.jsonl,${AH}_gpt4o_query.jsonl,${AH}_last_turn_query.jsonl,/scratch/gpfs/hyen/data/simpleqa/simple_qa_test_set.csv"
QUERIES="query,query,query,query,query,query,query,query,query,problem"
OUTPUT_DIR="outputs/$(basename $MODEL)"
OUTPUTS="$OUTPUT_DIR/alpaca_eval_davinci_query.jsonl,$OUTPUT_DIR/alpaca_eval_gpt4o_query.jsonl,$OUTPUT_DIR/alpaca_eval_instruction_query.jsonl,$OUTPUT_DIR/wild_bench_all_turns_query.jsonl,$OUTPUT_DIR/wild_bench_gpt4_query.jsonl,$OUTPUT_DIR/wild_bench_intent_query.jsonl,$OUTPUT_DIR/wild_bench_last_turn_query.jsonl,$OUTPUT_DIR/arena_hard_gpt4o_query.jsonl,$OUTPUT_DIR/arena_hard_last_turn_query.jsonl,$OUTPUT_DIR/simple_qa_test_set.jsonl"

EMB_FILES="embeddings/gte-large-en-v1.5/dclm_baseline_[0-4]*.pkl"
EMB_FILES="embeddings/gte-large-en-v1.5/fineweb_edu/fineweb_edu_*.pkl"

TOPK=100

echo "Dataset               = $DATASETS"
echo "Emb                   = $EMB_FILES"
echo "Output file           = $OUTPUTS"

python passage_retrieval.py \
    --model_name_or_path $MODEL \
    --embeddings "$EMB_FILES" \
    --dataset $DATASETS \
    --query_field $QUERIES \
    --output_file $OUTPUTS \
    --topk $TOPK \
    --query_max_length 8192 \
    --encode_batch_size 32 \
    --index_batch_size 8192

wait;

# Finish the script
