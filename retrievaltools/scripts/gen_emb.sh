#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=emb ## CHANGE JOBNAME HERE
#SBATCH --array=0-499%80

# Remove one # to uncommment
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=0-6:00:00
#SBATCH -x "della-i14g[1-20]"
##SBATCH -x /home/tianyug/pli_node_k
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
#mamba activate f1
mamba activate xform

IDX=$SLURM_ARRAY_TASK_ID
if [[ -z $IDX ]]; then
    IDX=1
    echo "IDX = $IDX"
fi

# Submit jobs.
# You can use srun to run multiple scripts in the same job in parallel (make sure to use & at the end!). Note how you can specify the resources used for each srun and make them exclusive using the --exclusive flag.
#srun --gres=gpu:1 -n 1 --mem=24G --exclusive python scripts/train_model.py --model_type ${model} --learning_rate ${lr} --rnn_dropout ${dropout} --checkpoint_path ${OUT_DIRECTORY}/${model}.lr${lr}.dim${dim}.dropout${dropout}.pt --rnn_wordvec_dim ${dim} --rnn_num_layers 2  --tags ${tag} &

# data_path="/scratch/gpfs/hyen/data/kilt/psgs_w100.tsv"
# data_path="/scratch/gpfs/hyen/data/kilt/kilt_wikipedia.jsonl"

PREFIX="dclm_baseline_dedup_256"
N=500

# PREFIX="wikipedia_dpr"
# PREFIX="wikipedia_kilt"
# N=10

# PREFIX="fineweb_edu_256"
# N=250

MODEL="gte-Qwen2-1.5B-instruct"

# OUTPUT_DIR="embeddings/$(basename $MODEL)/$PREFIX"
OUTPUT_DIR="/scratch/gpfs/DANQIC/hyen/embeddings/$(basename $MODEL)/$PREFIX"

echo "OUTPUT_DIR        = $OUTPUT_DIR"
echo "MODEL             = $MODEL"
echo "PREFIX            = $PREFIX"
echo "IDX               = $IDX / $N"

python generate_passage_embeddings.py \
    --config configs/models/$MODEL.yaml configs/corpus/$PREFIX.yaml \
    --output_dir $OUTPUT_DIR \
    --output_prefix $PREFIX  \
    --shard_id $IDX --num_shards $N \
    --input_max_length 512 \
    --batch_size 256 \
    --save_text

wait;

# Finish the script
#exit
