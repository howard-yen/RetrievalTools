import os
import argparse
import json
import logging
import glob
import yaml
from typing import Dict, List, Union, Any
from dataclasses import asdict

from tqdm import tqdm
import numpy as np
from simple_parsing import ArgumentParser

from arguments import (
    ModelOptions,
    IndexOptions,
    RetrievalDataOptions,
)
from index import DenseIndex # FAISS needs to be imported before pandas for some reason
from encoder import Encoder, load_encoder
from datasets import load_dataset
import datatools

import pandas as pd

import logging
logger = logging.getLogger(__name__)


class QueryList:
    """
    Supports mapping between flattened and unflattened queries.
    """
    def __init__(self, queries: Dict[str, List[Union[List[str], str]]]):
        # Queries map from the dataset name (a string) to a list (all samples in the dataset)
        # The list contains the queries for each sample, and each sample may have multiple queries
        # So the list elements may be either a list of strings or just one string
        self.queries = queries
        self.total_length = len(self.get_flattened())
    
    def get_flattened(self) -> List[str]:
        """
        Return all queries as a flattened list of strings
        """
        all_queries = []
        for _, dataset in self.queries.items():
            for example in dataset:
                all_queries += example if isinstance(example, list) else [example]
        return all_queries 

    def get_unflattened(self, flattened_results: List[Any]):
        """
        Given the flattened results, map them back to the original structure.
        
        Args:
            flattened_results: List of results corresponding to flattened queries
            
        Returns:
            Dictionary mapping dataset names to lists of results.
            For multi-query examples, returns a dictionary mapping queries to their results.
            For single-query examples, returns the result directly.
            
        Raises:
            ValueError: If flattened_results length doesn't match total number of queries
        """
        if len(flattened_results) != self.total_length:
            raise ValueError(f"Flattened results length ({len(flattened_results)}) does not match total number of queries ({self.total_length})")

        output = {}
        i = 0
        for dataset, queries in self.queries.items():
            output[dataset] = []
            for query in queries:
                if isinstance(query, list):
                    result_dict = {q: flattened_results[i+j] for j, q in enumerate(query)}
                    i += len(query)
                else:
                    result_dict = flattened_results[i]
                    i += 1
                output[dataset].append(result_dict)

        return output


def get_query(data, query_field: str, dataset: str):
    # if you need to get the query from any other nested structure, add it here
    # but I would recommend just preprocessing the data to have the query in the top level
    # if "arena-hard" in dataset:
    #     return [d['turns'][-1]['content'] for d in data]

    return [d[query_field] for d in data]


def main(args):
    all_data = {}
    all_queries = {}
    for dataset_options in args.data_options.datasets:
        data = datatools.load(dataset_options.input_path)
        all_data[dataset_options.input_path] = data
        all_queries[dataset_options.input_path] = get_query(data, dataset_options.query_field, dataset_options.input_path)
    query_list = QueryList(all_queries)

    logger.info("Loading retriever and index")
    encoder = load_encoder(args.model_options)
    index = DenseIndex(
        encoder=encoder,
        dtype="float16" if not args.index_options.no_fp16 else "float32",
        n_subquantizers=args.index_options.n_subquantizers,
        n_bits=args.index_options.n_bits,
    )

    logger.info("Encoding queries")
    query_emb = encoder.encode(query_list.get_flattened(), prompt=args.model_options.query_prompt)
    all_results = [{"scores": [], "ids": [], "texts": []} for _ in range(query_emb.shape[0])]

    logger.info("Searching through embeddings")
    emb_files = glob.glob(args.embeddings)
    total_passages = 0
    tbar = tqdm(emb_files)
    # we first retrieve the topk results from each embedding file before sorting
    # this is to reduce memory usage, but it could be optimized by loading multiple files at a time
    # one potential problem is that some embedding files may have texts but others don't, they need to be consistent so that the sorting is correct
    text_encountered = False
    for file_idx, file in enumerate(tbar):
        index.build_index(emb_files=[file])
        results = index.get_topk(query_emb=query_emb, topk=args.data_options.topk)

        for i, result in enumerate(results):
            ids = result["id"]
            scores = result["score"]
            texts = result.get("text", None)

            # only keep the topk results after searching through each embedding file
            new_ids = all_results[i]["ids"] + ids
            new_scores = all_results[i]["scores"] + scores.tolist()
            topk = np.argsort(-np.array(new_scores))[:args.data_options.topk]
            all_results[i]["ids"] = [new_ids[j] for j in topk]
            all_results[i]["scores"] = [new_scores[j] for j in topk]
            if texts is not None:
                assert text_encountered or file_idx == 0, "Some embedding files have texts but others don't"
                text_encountered = True
                new_texts = all_results[i]["texts"] + texts
                all_results[i]["texts"] = [new_texts[j] for j in topk]
            else:
                assert not text_encountered, "Some embedding files have texts but others don't"

        # another option is to load all the emb files without resetting the index
        total_passages += index.index_size
        tbar.set_description(f"Total passages indexed: {total_passages:,}")
        index.reset_index() 
    
    for i, result in enumerate(all_results):
        ids = result["ids"]
        scores = result["scores"]
        # sort the results by scores
        topk = np.argsort(-np.array(scores))[:args.data_options.topk]
        result["ids"] = [ids[j] for j in topk]
        result["scores"] = [scores[j] for j in topk]
        if result["texts"] != []:
            result["texts"] = [result["texts"][j] for j in topk]

    logger.info(f"Total passages indexed: {total_passages}")
    # now we need to map the results back to the original structure
    results = query_list.get_unflattened(all_results)
    for idx, (dataset, data) in enumerate(all_data.items()):
        dataset_options = args.data_options.datasets[idx]
        if dataset_options.output_path is None:
            output_file = os.path.join(args.output_dir, os.path.splitext(os.path.basename(dataset))[0] + ".jsonl")
        else:
            output_file = dataset_options.output_path

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        logger.info(f"Writing results to {output_file}")
        with open(output_file, "w") as f:
            for i, (d, res) in enumerate(zip(data, results[dataset])):
                if len(res['texts']) > 0:
                    d['ctxs'] = [{"id": i, "score": s, **t} for i, s, t in zip(res['ids'], res['scores'], res['texts'])]
                else:
                    d['ctxs'] = [{"id": i, "score": s} for i, s in zip(res['ids'], res['scores'])]
                f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    """
    This script retrieves passages from (possibly sharded) embedding files for a set of (possibly sharded) queries.    
    """
    parser = ArgumentParser(add_config_path_arg=True)

    parser.add_arguments(ModelOptions, dest="model_options")
    parser.add_arguments(IndexOptions, dest="index_options")
    parser.add_arguments(RetrievalDataOptions, dest="data_options")

    parser.add_argument("--embeddings", type=str, default=None, help="Glob path to embedding files (expected to be in .pkl with (ids, embeddings, [text optional]) tuples)", required=True)
    parser.add_argument("--overwrite", action="store_true", help="overwrite output path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()
    main(args)
    