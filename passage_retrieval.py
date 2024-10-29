
# this needs to be loaded first?

import os
import argparse
import json
import logging
import glob
import yaml

from tqdm import tqdm
import numpy as np

# note: need to load faiss before datatools?
from retriever import load_retriever
from datasets import load_dataset
from datatools import load

import logging
logger = logging.getLogger(__name__)


def load_data(dataset: str):
    # three possibilities: a single file, a glob path, or a huggingface path
    if os.path.isfile(dataset):
        data = load(dataset)

    elif "*" in dataset:
        data = load(*glob.glob(dataset))

    else:
        if dataset == "alpaca_eval":
            data = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

        elif dataset == "wild_bench":
            data = load_dataset("allenai/WildBench", "v2", split="test")

        else:
            data = load_dataset(dataset)

    return data


def get_query(data, query_field: str, dataset: str):
    # this is hacky, but idk 
    if "arena-hard" in dataset:
        return [d['turns'][-1]['content'] for d in data]
    elif 'wild_bench' in dataset:
        return [d['conversation_input'][-1]['content'] for d in data]
    else:
        return [d[query_field] for d in data]


def main(args):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    logger.info("Loading data and retriever")
    data = load_data(args.dataset)
    retriever = load_retriever(
        args.model_name_or_path,
        max_length=args.query_max_length,
        batch_size=args.per_gpu_batch_size,
        use_hf=args.use_hf,
    )

    # add support for encoding more queries 
    logger.info("Encoding queries")
    query_emb = retriever.encode(get_query(data, args.query_field, args.dataset), prompt=args.query_prompt)
    query_emb = retriever.encode(get_query(data, args.query_field, args.dataset), prompt=args.query_prompt)
    all_results = [([], []) for _ in range(query_emb.shape[0])]

    logger.info("Searching through embeddings")
    emb_files = glob.glob(args.embeddings)
    total_passages = 0
    tbar = tqdm(emb_files)
    for file in tbar:
        # we can load multiple files at once but maybe not necessary for now? maybe we should parallelize this? 
        retriever.build_index(emb_files=[file])
        results = retriever.get_topk(query_emb=query_emb, topk=args.topk)
        # maybe only keep the top k after each iteration instead of sorting it all at the end? 
        for i, (ids, scores) in enumerate(results):
            all_results[i][0].extend(ids)
            all_results[i][1].extend(scores.tolist())

        # another option is to load all the emb files without resetting the index
        total_passages += retriever.index_size
        tbar.set_description(f"Total passages indexed: {total_passages}")
        retriever.reset_index() 
    
    for i, (ids, scores) in enumerate(all_results):
        topk = np.argsort(-np.array(scores))[:args.topk]
        all_results[i] = [(ids[j], scores[j]) for j in topk]
    
    logger.info(f"Total passages indexed: {total_passages}")
    logger.info(f"Writing results to {args.output_file}")
    # assume jsonl file for now, but should support others in the future
    with open(args.output_file, "w") as f:
        for d, results in zip(data, all_results):
            d["ctxs"] = [{"id": doc_id, "score": score} for doc_id, score in results]
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    """
    This script retrieves passages from (possibly sharded) embedding files for a set of (possibly sharded) queries.
    
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        default=None,
        help="Data file or directory containing data files (.tsv or .jsonl)",
    )
    parser.add_argument("--embeddings", type=str, default=None, help="Glob path to embedding files (expected to be in .pkl with (ids, embeddings) tuples)", required=True)
    parser.add_argument("--query_field", type=str, default="question", help="field in the data containing the question")
    parser.add_argument(
        "--output_file", type=str, default=None, help="Output file"
    )
    parser.add_argument("--topk", type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument("--shards", type=int, default=1, help="Number of shards to split the index files into (not necessary if the embedding files are already sharded?)")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )

    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--use_hf", action="store_true", help="use HuggingFace instead of SentenceTransformers")
    parser.add_argument("--query_max_length", type=int, default=512, help="Maximum number of tokens to encode in the query")
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    # parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    # parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    # parser.add_argument("--normalize_emb", action="store_true", help="normalize the embeddings")
    parser.add_argument("--overwrite", action="store_true", help="overwrite output path")

    parser.add_argument("--query_prompt", type=str, default="", help="text prepended to the query")
    parser.add_argument("--config", type=str, default=None, help="config file")

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    main(args)
