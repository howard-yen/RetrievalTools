# This code is adopted from Contriever: https://github.com/facebookresearch/contriever
import os

import argparse
import pickle
import yaml
from glob import glob

import torch
import torch.nn.functional as F

from tqdm import tqdm
from utils import init_logger
from retriever import load_retriever
from data import Corpus

import logging
logger = logging.getLogger(__name__)


def load_data(corpus: str, shard_id: int, num_shards: int):
    is_file = os.path.isfile(corpus)
    if not is_file:
        # expect a glob expression and shard on the file level
        all_files = sorted(glob(corpus))
        assert len(all_files) > 0, f"No files found with {corpus}"
        assert num_shards <= len(all_files), f"Number of shards {num_shards} is greater than number of files {len(all_files)}"

        # TODO: replace the logic with utils.shard_idx
        shard_size = len(all_files) // num_shards
        start_idx = shard_id * shard_size
        end_idx = start_idx + shard_size
        if shard_id == num_shards - 1:
            end_idx = len(all_files)
        
        data = Corpus(*all_files[start_idx:end_idx])
        chunks = data.get_chunks()
        ids, texts = list(chunks.keys()), list(chunks.values())
        
    else:
        data = Corpus(corpus)
        # expect a single and shard within the file
        chunks = data.get_chunks()
        ids, texts = list(chunks.keys()), list(chunks.values())

        shard_size = len(chunks) // num_shards
        start_idx = shard_id * shard_size
        end_idx = start_idx + shard_size
        if shard_id == num_shards - 1:
            end_idx = len(chunks)
        
        ids = ids[start_idx:end_idx]
        texts = texts[start_idx:end_idx]

    return ids, texts


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    save_file = os.path.join(args.output_dir, f"{args.output_prefix}_{args.shard_id:03d}.pkl")
    if os.path.exists(save_file):
        logger.info(f"File {save_file} already exists, skipping.")
        return

    ids, texts = load_data(args.corpus, args.shard_id, args.num_shards)
    logger.info(f"Loaded {len(ids)} passages")
    retriever = load_retriever(
        args.model_name_or_path, 
        max_length=args.passage_max_length, 
        batch_size=args.per_gpu_batch_size,
        use_hf=args.use_hf,
    )
    logger.info("Encoding...")
    emb = retriever.encode(texts, prompt=args.passage_prompt)

    with open(save_file, mode="wb") as f:
        pickle.dump((ids, emb), f)
    logger.info(f"Saved {len(ids)} passage embeddings to {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")

    parser.add_argument("--corpus", type=str, default=None, help="Path to corpus (.tsv or .jsonl file or a directory with shards with a glob expression)")
    parser.add_argument("--output_dir", type=str, default="wikipedia_embeddings", help="dir path to save embeddings")
    parser.add_argument("--output_prefix", type=str, default="passages", help="output path prefix to save embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument("--passage_max_length", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument("--num_workers", type=int, default=8, help="Maximum number of tokens in a passage")
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--use_hf", action="store_true", help="use HF instead of SentenceTransformers")
    # parser.add_argument("--no_title", action="store_true", help="title not added to the passage body")
    # parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    # parser.add_argument("--normalize_text", action="store_true", help="lowercase text before encoding")
    # parser.add_argument("--normalize_emb", action="store_true", help="normalize the embeddings")

    parser.add_argument("--passage_prompt", type=str, default="", help="text prepended to the passage")

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    main(args)