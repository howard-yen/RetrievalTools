import os
import sys
import json
import re
import string
import requests
import time
import numpy as np
import torch
from typing import Tuple, List, Dict
from rouge_score import rouge_scorer
from urllib.parse import urlparse
from tqdm.contrib.concurrent import thread_map
import bm25s
import Stemmer
from diskcache import Cache
from functools import lru_cache

import logging

CACHE_PATH = os.environ.get("RT_CACHE_PATH", "./cache")
cache = Cache(CACHE_PATH)

# Suppress crawl4ai logging
logging.getLogger("crawl4ai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("playwright").setLevel(logging.WARNING)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

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


def detect_content_type(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.path.lower().endswith('.pdf'):
        return 'pdf'

    try:
        response = requests.head(url, headers=HEADERS, timeout=(3, 5))
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        return "pdf" if "pdf" in content_type else "html"
    except Exception as e:
        return "html"


def find_snippet(texts: List[str], snippet: str, num_characters: int = 4000, scoring_func: str = "rouge"):
    """
    We iterate through the texts, break them into chunks of 1000 characters, and then use the scoring function to find the best chunk.
    The text is already split into arbitrary chunks.
    The scoring function can be "rouge" or "bm25".
    We also take the surrounding text of the snippet to fill up the num_characters.
    """
    assert scoring_func in ["rouge", "bm25"], "Scoring function must be either 'rouge' or 'bm25'"
    positions = []
    start = 0
    best_recall = 0
    best_idx = 0

    if scoring_func == 'bm25':
        stemmer = Stemmer.Stemmer('english')
        corpus_tokens = bm25s.tokenize(texts, stopwords='en', stemmer=stemmer)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        query_tokens = bm25s.tokenize(snippet, stopwords='en', stemmer=stemmer)
        results, scores = retriever.retrieve(query_tokens, k=1)
        best_idx = int(results[0, 0])

    for i, text in enumerate(texts):
        if scoring_func == "rouge":
            score = scorer.score(target=snippet, prediction=text)
            recall = score['rougeL'].recall
            if recall > best_recall:
                best_recall = recall
                best_idx = i
        positions.append((start, start + len(text)))
        start += len(text) + 1

    best_len = len(texts[best_idx])
    num_characters = num_characters - best_len
    final_text = []
    for i, pos in enumerate(positions):
        if (pos[0] >= positions[best_idx][0] - num_characters/2 and pos[1] <= positions[best_idx][1] + num_characters/2) or i == best_idx:
            final_text.append(texts[i])

    return "\n".join(final_text)


async def scrape_pdf(url: str) -> Tuple[bool, str, str]:
    import fitz
    response = requests.get(url, headers=HEADERS, timeout=(3, 5))  # (connect timeout, read timeout)
    response.raise_for_status()
    with fitz.open(stream=response.content, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    texts = text.split("\n")

    return True, text, text


async def scrape_html(url: str) -> Tuple[bool, str, str]:
    prune_filter = PruningContentFilter(threshold=0.4, threshold_type="dynamic", min_word_threshold=3)
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter, options={"ignore_links": False})
    browser_config = BrowserConfig(
        headless=True, verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox", "--disable-extensions"]
    )
    crawler_config = CrawlerRunConfig(markdown_generator=md_generator, page_timeout=15000, verbose=False)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await asyncio.wait_for(crawler.arun(url=url, config=crawler_config), timeout=30)

    if not result.success:
        return False, f"Failed to scrape the page due to {result.error_message}", ""

    if len(result.markdown.raw_markdown.strip()) == 0:
        return False, f"Failed to scrape the page, returned empty content.", ""

    fit_markdown = result.markdown.fit_markdown
    raw_markdown = result.markdown.raw_markdown

    return True, fit_markdown, raw_markdown

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 0.2


@lru_cache(maxsize=8192)
@cache.memoize(typed=True, expire=1e7, tag="serper")
def serper_search(query: str, topk: int = 10) -> List[Dict]:
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": os.environ["SERPER_API_KEY"], 'Content-Type': 'application/json'}
    payload = json.dumps({"q": query, "num": topk})

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url=url, headers=headers, data=payload)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                time.sleep(delay)
                continue
            else:
                raise e
        except Exception as e:
            raise e

        if response.status_code in [408, 500, 502, 503, 504]:
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                time.sleep(delay)
                continue
            else:
                raise Exception(response.text)

    response = response.json()
    return response

