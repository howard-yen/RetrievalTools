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
# mamba activate xform

IDX=$SLURM_ARRAY_TASK_ID
if [[ -z $IDX ]]; then
    IDX=1
    echo "IDX = $IDX"
fi

# Submit jobs.

PREFIX="wikipedia_dolma"
N=1000

# PREFIX="wikipedia_dpr"
# PREFIX="wikipedia_kilt"
# N=10

# PREFIX="fineweb_edu_256"
# N=250

MODEL="Qwen3-0.6B"

# OUTPUT_DIR="embeddings/$(basename $MODEL)/$PREFIX"
OUTPUT_DIR="/home/hyen/project/embeddings/$(basename $MODEL)/$PREFIX-test"

echo "OUTPUT_DIR        = $OUTPUT_DIR"
echo "MODEL             = $MODEL"
echo "PREFIX            = $PREFIX"
echo "IDX               = $IDX / $N"

# for IDX in {0..9}; do

python generate_passage_embeddings.py \
    --config configs/models/$MODEL.yaml configs/corpus/$PREFIX.yaml \
    --output_dir $OUTPUT_DIR \
    --output_prefix $PREFIX  \
    --shard_id $IDX --num_shards $N \
    --input_max_length 2048 \
    --save_text --overwrite --batch_size 32

# done

wait;

# Finish the script
exit
