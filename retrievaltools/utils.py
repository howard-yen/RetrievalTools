import os
import sys
import json
import re
import string
import requests
import time
import numpy as np
import torch
from typing import Tuple, List
from rouge_score import rouge_scorer
from urllib.parse import urlparse
from tqdm.contrib.concurrent import thread_map

import logging
import errno
import fcntl

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


def find_snippet(texts: List[str], snippet=None, num_characters=4000):
    # find the best snippet in the text that matches the snippet
    # take the surrounding text of the snippet
    if snippet is None:
        return "\n".join(texts)[:num_characters]

    # we iterate through the texts, calculate the ROUGE score between the snippet and the text
    # we mainly care about the recall score of ROUGE-L (most of the snippet is present in the long text)
    # take the text with the highest recall score and the surrounding text of the snippet

    positions = []
    start = 0
    best_recall = 0
    best_idx = 0
    for i, text in enumerate(texts):
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
    

# adapted from https://github.com/idavidrein/gpqa/blob/56686c06f5e19865c153de0fdb11be3890014df7/baselines/open_book.py#L38
def scrape_html(response, snippet=None, num_characters=4000) -> Tuple[bool, str, str]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Remove unwanted elements
    for element in soup.find_all(['script', 'style', 'nav', 'footer', 'iframe']):
        element.decompose()

    # Keep track of character positions for each element
    texts = []
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
        text = element.text.strip()
        # Add header markers for better context
        if element.name.startswith('h'):
            text = f"[{element.name.upper()}] {text}"
        
        if len(text) == 0:
            continue
        
        texts.append(text)

    # Join all text for snippet matching
    fulltext = "\n".join(texts)
    final_snippet = find_snippet(texts, snippet, num_characters)

    return True, final_snippet, fulltext


def scrape_pdf(response, snippet=None, num_characters=4000) -> Tuple[bool, str, str]:
    import fitz
    with fitz.open(stream=response.content, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    
    texts = text.split("\n")
    final_snippet = find_snippet(texts, snippet, num_characters)
    return True, final_snippet, text


def scrape_page_content(url, snippet=None, num_characters=10000) -> Tuple[bool, str, str]:
    """
    Try to scrape the page content and return the best snippet that matches the snippet.
    If no snippet is provided, return the first num_characters of the page.
    Returns a tuple of (success, snippet, fulltext).
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=(3, 5))  # (connect timeout, read timeout)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if "pdf" in content_type:
            return scrape_pdf(response, snippet, num_characters)
        else:
            return scrape_html(response, snippet, num_characters)

    except Exception as e:
        success = False
        return success, f"Failed to scrape the page due to {str(e)}", ""


def detect_content_type(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.path.lower().endswith('.pdf'):
        return 'pdf'

    try:
        response = requests.head(url, headers=HEADERS, timeout=(3, 5))
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if "pdf" in content_type:
            return "pdf"
        else:
            return "html"
    except Exception as e:
        return "html"


async def scrape_page_content_crawl4ai_batch(urls: List[str], snippets: List[str] = None, verbose: bool = False) -> Tuple[bool, str, str]:
    from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
    from crawl4ai import CrawlerRunConfig, AsyncWebCrawler, CrawlerMonitor, RateLimiter, BrowserConfig
    
    prune_filter = PruningContentFilter(
        threshold=0.3,
        threshold_type="fixed",  
        min_word_threshold=3
    )

    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=85.0,
        check_interval=2.0,
        max_session_permit=5,
        fairness_timeout=120,
        rate_limiter=RateLimiter(
            base_delay=(0.5, 1.0),
            max_delay=5.0,
            max_retries=3
        ),
    )

    md_generator = DefaultMarkdownGenerator(
        content_filter=prune_filter,
        options={"ignore_links": False}
    )

    browser_config = BrowserConfig(
        headless=True,
        verbose=verbose,  # Suppress crawl4ai browser logging
        extra_args=[
          "--disable-gpu",
          "--disable-dev-shm-usage",
          "--no-sandbox",
          "--disable-extensions",
          "--disable-background-timer-throttling",
          "--disable-backgrounding-occluded-windows",
          "--disable-renderer-backgrounding"
        ]
    )

    def process_result(result, i, snippets):
        if result is None or not result.success:
            return (False, f"Failed to scrape the page due to {result.error_message}", "")
        else:
            fit_markdown = result.markdown.fit_markdown
            if snippets is not None:
                snippet = snippets[i]
                if len(fit_markdown) > 10000:
                    fit_markdown = find_snippet(fit_markdown.split("\n"), snippet, 10000)
            return (True, fit_markdown, result.markdown.raw_markdown)

    try:
        content_types = thread_map(detect_content_type, urls, desc="Detecting content types", disable=not verbose)
        pdf_urls = [url for url, content_type in zip(urls, content_types) if content_type == "pdf"]
        pdf_idxs = [i for i, content_type in enumerate(content_types) if content_type == "pdf"]
        html_urls = [url for url, content_type in zip(urls, content_types) if content_type == "html"]
        html_idxs = [i for i, content_type in enumerate(content_types) if content_type == "html"]
        all_results = [None for _ in range(len(urls))]

        if len(pdf_urls) > 0:
            # async with AsyncWebCrawler(config=browser_config, crawler_strategy=PDFCrawlerStrategy()) as crawler:
            #     results = await crawler.arun_many(
            #         urls=pdf_urls,
            #         config=CrawlerRunConfig(
            #             markdown_generator=md_generator,
            #             scraping_strategy=PDFContentScrapingStrategy(),
            #             page_timeout=10,
            #             exclude_external_images=True,
            #         ),
            #         dispatcher=dispatcher,
            #         only_text=True,
            #     )
            # for i, result in enumerate(results):
            #     all_results[pdf_idxs[i]] = process_result(result, i, snippets)

            results = thread_map(scrape_page_content, pdf_urls, [snippets[i] for i in pdf_idxs], desc="Scraping PDFs", disable=not verbose)
            for i, result in enumerate(results):
                all_results[pdf_idxs[i]] = results[i]

        if len(html_urls) > 0:
            try:
                import asyncio
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    results = await asyncio.wait_for(
                        crawler.arun_many(
                            urls=html_urls,
                            config=CrawlerRunConfig(
                                markdown_generator=md_generator,
                                page_timeout=15000,
                                exclude_external_images=True,
                                word_count_threshold=3,
                                verbose=verbose,  # Suppress crawl4ai run logging
                            ),
                            dispatcher=dispatcher,
                        ),
                        timeout=300
                    )
                for i, result in enumerate(results):
                    all_results[html_idxs[i]] = process_result(result, i, snippets)
            except (asyncio.TimeoutError, Exception) as e:
                print(f"Crawl4ai batch processing failed: {e}. Falling back to individual requests.")
                for i, url in enumerate(html_urls):
                    try:
                        snippet = snippets[html_idxs[i]] if snippets else None
                        result = await asyncio.wait_for(
                            scrape_page_content_crawl4ai(url, snippet, verbose=verbose),
                            timeout=30
                        )
                        all_results[html_idxs[i]] = result
                    except Exception:
                        fallback_result = scrape_page_content(url, snippet if snippets else None)
                        all_results[html_idxs[i]] = fallback_result

        return all_results

    except Exception as e:
        raise e
        return None


async def scrape_page_content_crawl4ai(url: str, snippet=None, verbose: bool = False) -> Tuple[bool, str, str]:
    from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
    from crawl4ai import CrawlerRunConfig, AsyncWebCrawler, BrowserConfig
    import asyncio

    prune_filter = PruningContentFilter(
        threshold=0.4,
        threshold_type="dynamic",  
        min_word_threshold=3
    )

    md_generator = DefaultMarkdownGenerator(
        content_filter=prune_filter,
        options={"ignore_links": False}
    )

    browser_config = BrowserConfig(
        headless=True,
        verbose=verbose,  # Suppress crawl4ai browser logging
        extra_args=[
            "--disable-gpu",
            "--disable-dev-shm-usage", 
            "--no-sandbox",
            "--disable-extensions",
        ]
    )

    try:
        content_type = detect_content_type(url)
        if content_type == "pdf":
            # async with AsyncWebCrawler(config=browser_config, crawler_strategy=PDFCrawlerStrategy()) as crawler:
            #     result = await asyncio.wait_for(
            #         crawler.arun(
            #             url=url,
            #             config=CrawlerRunConfig(
            #                 markdown_generator=md_generator,
            #                 scraping_strategy=PDFContentScrapingStrategy(),
            #                 page_timeout=15000,
            #                 verbose=verbose,  # Suppress crawl4ai run logging
            #             )
            #         ),
            #         timeout=30
            #     )
            result = scrape_page_content(url, snippet, 10000)
            return result
        else:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await asyncio.wait_for(
                    crawler.arun(
                        url=url,
                        config=CrawlerRunConfig(
                            markdown_generator=md_generator,
                            page_timeout=15000,
                            verbose=verbose,  # Suppress crawl4ai run logging
                        )
                    ),
                    timeout=30
                )
        if not result.success:
            return False, f"Failed to scrape the page due to {result.error_message}", ""

        fit_markdown = result.markdown.fit_markdown 
        raw_markdown = result.markdown.raw_markdown

        # if the markdown is too long, we truncate it
        if len(fit_markdown) > 10000 and snippet is not None:
            fit_markdown = find_snippet(fit_markdown.split("\n"), snippet, 10000)

        return True, fit_markdown, raw_markdown

    except Exception as e:
        return False, f"Failed to scrape the page due to {str(e)}", ""


class ThreadSafeFileHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        
    def _acquire_lock(self, file_obj):
        try:
            fcntl.flock(file_obj, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError as e:
            if e.errno != errno.EAGAIN:
                raise
            # If file is locked, retry with exponential backoff
            retries = 5
            base_wait_time = 0.1
            for i in range(retries):
                time.sleep(base_wait_time * (2 ** i))
                try:
                    fcntl.flock(file_obj, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except IOError as e:
                    if e.errno != errno.EAGAIN:
                        raise
                    continue
            return False
        return True
    
    def _release_lock(self, file_obj):
        fcntl.flock(file_obj, fcntl.LOCK_UN)

    def _ensure_file_exists(self):
        if not os.path.exists(self.filepath):
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'w') as f:
                json.dump({}, f)
    
    def read_data(self):
        self._ensure_file_exists()
        
        with open(self.filepath, 'r+') as file_obj:
            if not self._acquire_lock(file_obj):
                raise TimeoutError("Could not acquire lock for reading")
            
            try:
                file_obj.seek(0)
                data = json.load(file_obj)
                return data
            finally:
                self._release_lock(file_obj)
    
    def update_data(self, update_func):
        """
        Update the file using the provided update function.
        update_func should take the current data as input and return the updated data.
        """
        self._ensure_file_exists()
        
        with open(self.filepath, 'r+') as file_obj:
            if not self._acquire_lock(file_obj):
                raise TimeoutError("Could not acquire lock for updating")
            
            try:
                file_obj.seek(0)
                try:
                    current_data = json.load(file_obj)
                except json.JSONDecodeError:
                    current_data = {}
                
                # Apply the update function
                updated_data = update_func(current_data)
                
                # Write the updated data back to the file
                file_obj.seek(0)
                file_obj.truncate()
                json.dump(updated_data, file_obj)
                return updated_data
            finally:
                self._release_lock(file_obj)
