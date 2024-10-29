import os

import argparse 
import json

from tqdm import tqdm
from datasets import load_dataset

from data import Corpus

"""
We add the document texts using the ids from the corpus.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to the retrieval results file")
    args = parser.parse_args()

    data = load_dataset("json", data_files=args.data_file)['train']
    corpus = Corpus()

    def annotate(example):
        for ctx in tqdm(example['ctxs']):
            ctx['text'] = corpus.get_text_greedy(ctx['id'])
        return example
    
    annotated_data = data.map(annotate, num_proc=32)

    # save the annotated data as a jsonl file
    annotated_data.to_json(args.data_file.replace(".jsonl", "_annotated.jsonl"), orient="records", force_ascii=False)
    