import os
import sys

import re
import string

import time
import numpy as np
import torch
from typing import Tuple

import logging


def init_logger(name, args=None, stdout_only=False):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler]
    if args is not None and not stdout_only:
        file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, "run.log"))
        handlers.append(file_handler)
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(name)


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def cls_pooling(token_embeddings, mask):
    # first token where mask is 1
    indices = mask.argmax(1)
    return token_embeddings[torch.arange(token_embeddings.size(0)), indices]


def max_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), float('-inf'))
    sentence_embeddings, _ = token_embeddings.max(dim=1)
    return sentence_embeddings


def last_token_pooling(token_embeddings, mask):
    # find the last index of the mask (last occurance of a 1)
    indices = mask.size(1) - 1 - torch.fliplr(mask).argmax(dim=1)
    return token_embeddings[torch.arange(token_embeddings.size(0)), indices]


POOLING_FUNC = {
    "mean": mean_pooling,
    "cls": cls_pooling,
    "max": max_pooling,
    "last": last_token_pooling,
}

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def drqa_metric_max_over_ground_truths(metric_fn, prediction, ground_truths) -> int:
    """
    Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    # ground truth could be a string or a list of strings or a list of list of strings
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif isinstance(ground_truths[0], list):
        ground_truths = [ground_truth for ground_truths_list in ground_truths for ground_truth in ground_truths_list]

    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_shard_idx(size: int, num_shards: int, shard_id: int) -> Tuple[int, int]:
    # currently, we try to distribute the shards as evenly as possible
    # this could result in shard sizes like 2 1 2 1 ... 
    # another approach is to try to make all the shards the same size except for possibly the last one, which could be smaller
    # that would make the processing time for all but one shard the exact same
    shard_indices = np.linspace(0, size, num_shards+1, endpoint=True).astype(int)
    return shard_indices[shard_id], shard_indices[shard_id+1]

