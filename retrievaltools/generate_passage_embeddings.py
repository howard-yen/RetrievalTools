# This code is adopted from Contriever: https://github.com/facebookresearch/contriever
import os

import pickle
import yaml
from glob import glob
from simple_parsing import ArgumentParser
from dataclasses import asdict

from arguments import ModelOptions, CorpusOptions, ShardOptions
from utils import init_logger, get_shard_idx
from encoder import load_encoder
from data import Corpus, load_corpus

logger = init_logger(__name__)


def load_corpus(corpus_options: CorpusOptions, shard_options: ShardOptions):
    corpus = Corpus(
        paths=corpus_options.paths, 
        id_field=corpus_options.id_field, 
        text_field=corpus_options.text_field, 
        metadata_fields=corpus_options.metadata_fields, 
        num_workers=corpus_options.num_workers, 
        shard_options=shard_options
    )
    chunks = corpus.get_data()
    ids, texts = list(chunks.keys()), list(chunks.values())
    return ids, texts


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    save_file = os.path.join(args.output_dir, f"{args.output_prefix}_{args.shard_options.shard_id:04d}.pkl")
    if os.path.exists(save_file) and not args.overwrite:
        logger.info(f"File {save_file} already exists, skipping.")
        return

    ids, texts = load_corpus(args.corpus_options, args.shard_options)
    raw_texts = [texts['text'] for texts in texts]
    logger.info(f"Loaded {len(ids)} passages")
    encoder = load_encoder(args.model_options)
    logger.info("Encoding...")
    emb = encoder.encode(raw_texts, prompt=args.model_options.passage_prompt)

    with open(save_file, mode="wb") as f:
        if args.save_text:
            pickle.dump((ids, emb, texts), f)
        else:
            pickle.dump((ids, emb), f)
    logger.info(f"Saved {len(ids)} passage embeddings to {save_file}")

    # save the corpus args
    with open(os.path.join(args.output_dir, "corpus_args.yaml"), "w") as f:
        yaml.dump(asdict(args.corpus_options), f)
    logger.info(f"Saved corpus args to {os.path.join(args.output_dir, 'corpus_args.yaml')}")


if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(ModelOptions, dest="model_options")
    parser.add_arguments(CorpusOptions, dest="corpus_options")
    parser.add_arguments(ShardOptions, dest="shard_options")

    parser.add_argument("--output_dir", type=str, default="wikipedia_embeddings", help="dir path to save embeddings")
    parser.add_argument("--output_prefix", type=str, default="passages", help="output path prefix to save embeddings")
    parser.add_argument("--save_text", action="store_true", help="save the text and extra metadata to the output file (requires more disk but makes retrieval easier)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing embeddings")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    main(args)
