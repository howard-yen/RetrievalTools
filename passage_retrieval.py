import os
import argparse
import json
import logging
import glob
import yaml
from typing import Dict, List, Union, Any

from tqdm import tqdm
import numpy as np

from retriever import load_retriever
from index import DenseIndex # FAISS needs to be imported before pandas for some reason
from datasets import load_dataset
from datatools import load

import pandas as pd

import logging
logger = logging.getLogger(__name__)


class QueryList:
    """
    Supports mapping between flattened and unflattened queries.
    """
    def __init__(self, queries: Dict[Dict, List[Union[Dict, str]]]):
        # An example mapping is from Dict[dataset (str), List[Dict[query_field (str), query (str)]]]
        self.queries = queries
    
    def get_flattened(self):
        """
        Return all queries as a flattened list
        """
        output = []
        for _, dataset in self.queries.items():
            for example in dataset:
                output += list(example.values()) if isinstance(example, dict) else [example]
        return output

    def get_unflattened(self, flattened_results: List[Any]):
        """
        Given the flattened results, map them back to the original structure
        """
        output = {}
        i = 0
        for dataset, dataset_data in self.queries.items():
            output[dataset] = []
            for example in dataset_data:
                output[dataset].append(
                    {k: flattened_results[i] for k in example.keys()} 
                    if isinstance(example, dict) 
                    else flattened_results[i]
                )
                i += len(example) if isinstance(example, dict) else 1

        return output


def load_data(dataset: str):
    # three possibilities: a single file, a glob path, or a huggingface path
    if os.path.isfile(dataset):
        if dataset.endswith(".tsv"):
            data = pd.read_csv(dataset, sep="\t")
            data = [row.to_dict() for _, row in data.iterrows()]

        elif dataset.endswith(".csv"):
            data = pd.read_csv(dataset)
            data = [row.to_dict() for _, row in data.iterrows()]
        
        else:
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
    # if you need to get the query from any other nested structure, add it here
    # if "arena-hard" in dataset:
    #     return [d['turns'][-1]['content'] for d in data]
    # elif 'wild_bench' in dataset:
    #     return [d['conversation_input'][-1]['content'] for d in data]

    return [d[query_field] for d in data]


def main(args):
    assert len(args.dataset.split(",")) == len(args.query_field.split(",")), "Number of datasets and query fields must match"
    assert len(args.dataset.split(",")) == len(args.output_file.split(",")), "Number of datasets and output files must match"
    
    logger.info("Loading data")
    all_data = {}
    all_queries = {}
    for file, query_field in zip(args.dataset.split(","), args.query_field.split(",")):
        file = file.strip()
        query_field = query_field.strip()
        data = load_data(file)
        all_data[file] = data
        all_queries[file] = get_query(data, query_field, file)
    query_list = QueryList(all_queries)

    logger.info("Loading retriever and index")
    retriever = load_retriever(
        args.model_name_or_path,
        max_length=args.query_max_length,
        batch_size=args.encode_batch_size,
        use_hf=args.use_hf,
    )
    index = DenseIndex(
        encoder=retriever,
        dtype="float16" if not args.no_fp16 else "float32",
        n_subquantizers=args.n_subquantizers,
        n_bits=args.n_bits,
    )

    # add support for encoding more queries 
    logger.info("Encoding queries")
    query_emb = retriever.encode(query_list.get_flattened(), prompt=args.query_prompt)
    all_results = [([], []) for _ in range(query_emb.shape[0])]

    logger.info("Searching through embeddings")
    emb_files = glob.glob(args.embeddings)
    total_passages = 0
    tbar = tqdm(emb_files)
    for file in tbar:
        # we can load multiple files at once but maybe not necessary for now? maybe we should parallelize this? 
        index.build_index(emb_files=[file])
        results = index.get_topk(query_emb=query_emb, topk=args.topk)

        for i, (ids, scores) in enumerate(results):
            new_ids = all_results[i][0] + ids
            new_scores = all_results[i][1] + scores.tolist()
            topk = np.argsort(-np.array(new_scores))[:args.topk]
            all_results[i] = [new_ids[j] for j in topk], [new_scores[j] for j in topk]

        # another option is to load all the emb files without resetting the index
        total_passages += index.index_size
        tbar.set_description(f"Total passages indexed: {total_passages:,}")
        index.reset_index() 
    
    for i, (ids, scores) in enumerate(all_results):
        topk = np.argsort(-np.array(scores))[:args.topk]
        all_results[i] = [(ids[j], scores[j]) for j in topk]

    logger.info(f"Total passages indexed: {total_passages}")
    # now we need to map the results back to the original structure
    results = query_list.get_unflattened(all_results)
    for idx, (dataset, data) in enumerate(all_data.items()):
        output_file = args.output_file.split(",")[idx]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        logger.info(f"Writing results to {output_file}")
        with open(output_file, "w") as f:
            for i, (d, r) in enumerate(zip(data, results[dataset])):
                d["ctxs"] = [{"id": doc_id, "score": score} for doc_id, score in r]
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
    parser.add_argument("--encode_batch_size", type=int, default=64, help="Batch size for query encoding")
    parser.add_argument("--index_batch_size", type=int, default=2048, help="Batch size for index search")
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
