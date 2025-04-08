import os
import sys

import re
import string
import requests

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


def remove_punctuation(text):
    # Using a translation table to remove punctuation
    return text.translate(str.maketrans("", "", string.punctuation))


def f1_score(true_set, pred_set):
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)


# adapted from https://github.com/idavidrein/gpqa/blob/56686c06f5e19865c153de0fdb11be3890014df7/baselines/open_book.py#L38
def scrape_page_content(url, snippet=None, num_characters=4000) -> Tuple[bool, str, str]:
    from bs4 import BeautifulSoup
    """
    Try to scrape the page content and return the best snippet that matches the snippet.
    If no snippet is provided, return the first num_characters of the page.
    Returns a tuple of (success, snippet, fulltext).
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers, timeout=(3, 5))  # (connect timeout, read timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()

        # Keep track of character positions for each element
        content_elements_with_pos = []
        current_pos = 0
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
            text = element.text.strip()
            # Add header markers for better context
            if element.name.startswith('h'):
                text = f"[{element.name.upper()}] {text}"
            
            if len(text) == 0:
                continue
            
            # Store the text along with its position information
            content_elements_with_pos.append({
                'text': text,
                'start': current_pos,
                'end': current_pos + len(text)
            })
            current_pos += len(text) + 2  # +2 for the "\n\n" separator

        # Join all text for snippet matching
        fulltext = "\n\n".join(elem['text'] for elem in content_elements_with_pos)
        
        best_sentence = None
        # find which sentence overlaps with the snippet most in terms of f1 score
        if snippet is not None:
            snippet = snippet.lower()
            # remove punctuation
            snippet = remove_punctuation(snippet)
            snippet_words = set(snippet.split())
            best_f1 = 0
            sentences = fulltext.split(".")
            for sentence in sentences:
                key_sentence = sentence.lower()
                key_sentence = remove_punctuation(key_sentence)
                sentence_words = set(key_sentence.split())
                f1 = f1_score(snippet_words, sentence_words)
                if f1 > best_f1:
                    best_f1 = f1
                    best_sentence = sentence

        success = True
        if best_sentence is not None and len(best_sentence) > 0:
            # Find the position of the best matching sentence
            para_start = fulltext.find(best_sentence)
            para_end = para_start + len(best_sentence)

            # Find elements that should be included (within 2000 chars before and after)
            included_elements = []
            for elem in content_elements_with_pos:
                # Check if element is within range or contains the best sentence
                if (elem['start'] >= para_start - num_characters/2 and elem['start'] <= para_end + num_characters/2) or \
                   (elem['start'] <= para_start and elem['end'] >= para_end):
                    included_elements.append(elem['text'])
            
            final_snippet = "\n\n".join(included_elements)
            # prevent super long snippets due to parsing errors and such
            if len(final_snippet) > num_characters*2:
                final_snippet = final_snippet[:num_characters*2]
            return success, final_snippet, fulltext
        
        # If no best sentence, return first few complete elements up to ~4000 chars
        included_elements = []
        total_length = 0
        for elem in content_elements_with_pos:
            if total_length + len(elem['text']) > num_characters and total_length > 0:
                break
            included_elements.append(elem['text'])
            total_length += len(elem['text']) + 2  # +2 for "\n\n"
        final_snippet = "\n\n".join(included_elements)
        if len(final_snippet) > num_characters*2:
            final_snippet = final_snippet[:num_characters*2]
        return success, final_snippet, fulltext

    except Exception as e:
        success = False
        return success, f"Failed to scrape the page due to {str(e)}", ""

