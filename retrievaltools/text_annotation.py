import os
from typing import List, Dict

from tqdm import tqdm, trange
from datasets import load_dataset
from rouge_score import rouge_scorer
from simple_parsing import ArgumentParser

from retrievaltools.arguments import CorpusOptions, ShardOptions
from retrievaltools.data import load_corpus
from retrievaltools.utils import init_logging

logger = init_logging(__name__)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
def deduplication(ctxs: List[Dict[str, str]], threshold: int = 0.8, k: int = 5) -> List[Dict[str, str]]:
    texts = set()
    discarded = set()
    new_ctxs = []
    for ctx in tqdm(ctxs, leave=False):
        if len(texts) == 0:
            texts.add(ctx['text'])
            new_ctxs.append(ctx)
            continue
        elif len(texts) >= k:
            break
        if ctx['text'] in texts or ctx['text'] in discarded:
            discarded.add(ctx['text'])
            continue
        
        scores = scorer.score_multi(targets=texts, prediction=ctx['text'])
        if scores['rougeL'].fmeasure > threshold:
            discarded.add(ctx['text'])
            continue
        new_ctxs.append(ctx)
        # texts.append(ctx['text'])
        texts.add(ctx['text'])
    return new_ctxs


def get_output_file(data_file: str, shard_options: ShardOptions, anot=False, dedup=False) -> str:
    assert anot != dedup, "Can only annotate or deduplicate, not both"
    if shard_options.num_shards == 1:
        if anot:
            return data_file.replace(".jsonl", "_anot.jsonl")
        else:
            return data_file.replace(".jsonl", "_anot_dedup.jsonl")
    else:
        basedir = os.path.splitext(data_file)[0]
        os.makedirs(basedir, exist_ok=True)
        output = os.path.join(basedir, os.path.basename(data_file).replace(".jsonl", f"_{shard_options.shard_id}.jsonl"))
        if anot:
            return output.replace(".jsonl", f"_anot.jsonl")
        else:
            return output.replace(".jsonl", f"_anot_dedup.jsonl")
    

"""
We add the document texts using the ids from the corpus.
"""
if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(CorpusOptions, dest="corpus_options")
    parser.add_arguments(ShardOptions, dest="shard_options")

    parser.add_argument("--data_file", type=str, required=True, help="Path to the retrieval results file (should be a jsonl file)")
    parser.add_argument("--deduplicate", action="store_true", help="Deduplicate the data, using ROUGE-L")
    parser.add_argument("--deduplicate_threshold", type=float, default=0.9, help="Threshold for deduplication")
    parser.add_argument("--topk", type=int, default=100, help="Threshold for deduplication")
    parser.add_argument("--num_proc", type=int, default=os.cpu_count(), help="Number of processes to use")
    args = parser.parse_args()
    logger.info(args)
    assert args.data_file.endswith(".jsonl")

    logger.info("Loading corpus, make sure that the corpus settings are the same as the encoding!")
    corpus = load_corpus(args.corpus_options)

    def annotate(example):
        for ctx in tqdm(example['ctxs']):
            ctx['text'] = corpus.get_text_greedy(ctx['id'])
        return example

    data = load_dataset("json", data_files=args.data_file)['train'] 

    if args.shard_options.num_shards > 1:
        data = data.shard(num_shards=args.shard_options.num_shards, index=args.shard_options.shard_id)
    
    out_file = get_output_file(args.data_file, args.shard_options, anot=True)
    if os.path.exists(out_file):
        logger.info(f"File {out_file} already exists, skipping")
        annotated_data = load_dataset("json", data_files=out_file)['train']
    else:
        logger.info("Annotating")
        annotated_data = data.map(annotate, num_proc=args.num_proc, desc="Annotating")
        
        # save the annotated data as a jsonl file
        logger.info(f"Saving annotated data to {out_file}")
        annotated_data.to_json(out_file, orient="records", force_ascii=False)

    if args.deduplicate:
        def dedup(example):
            example['ctxs'] = deduplication(example['ctxs'], threshold=args.deduplicate_threshold, k=args.topk)
            return example

        out_file = get_output_file(args.data_file, args.shard_options, dedup=True)
        if os.path.exists(out_file):
            logger.info(f"File {out_file} already exists, skipping")
            annotated_data = load_dataset("json", data_files=out_file)['train']
        else:
            logger.info("Deduplicating")
            annotated_data = annotated_data.map(dedup, num_proc=args.num_proc, desc="Deduplicating")
            annotated_data.to_json(out_file, orient="records", force_ascii=False)
    
    # now collect all the shards if possible
    def concat(data_file: str, shard_options: ShardOptions, anot=False, dedup=False) -> bool:
        logger.info("Concatenating")
        for i in trange(shard_options.num_shards, desc="Checking files"):
            shard_options.shard_id = i
            file = get_output_file(data_file, shard_options, anot=anot, dedup=dedup)
            if not os.path.exists(file):
                logger.info(f"File {file} does not exist, skipping concat")
                return False
        
        # a bit hacky...
        n = shard_options.num_shards
        shard_options.num_shards = 1
        out_file = get_output_file(data_file, shard_options, anot=anot, dedup=dedup)
        shard_options.num_shards = n
        with open(out_file, "w") as fout:
            for i in trange(shard_options.num_shards, desc="Writing files"):
                shard_options.shard_id = i
                file = get_output_file(data_file, shard_options, anot=anot, dedup=dedup)
                with open(file) as fin:
                    for line in fin:
                        fout.write(line)
        logger.info(f"Written concat to {out_file}")
        return True

    if args.deduplicate and concat(args.data_file, args.shard_options, dedup=True):
        logger.info("Concatenated annotated and deduplicated data")
