import os
import json
from typing import Optional, List, Dict, Any, Union, Tuple
import random
import time
from dataclasses import asdict
import requests
import threading
import asyncio
from tqdm import tqdm

from retrievaltools.arguments import CorpusOptions, IndexOptions, ModelOptions, RetrieverOptions
from retrievaltools.utils import scrape_page_content, init_logger, ThreadSafeFileHandler, scrape_page_content_crawl4ai_batch

logger = init_logger(__name__)


class Retriever():
    """
    The retriever class is a simple interface that takes queries as inputs and return results. 

    In the results, we guarantee that "text" is present. Other fields are dependent on the specific retriever (see subclasses).

    We also support caching the results of the retrieval.
    The cache is a dictionary that maps queries to results, where the results are the same format as the output of the retrieve method.
    We recommend using the cache to speed up the retrieval process and also save and load the cache to disk.
    Make to handle race conditions when saving and loading the cache.
    """
    def __init__(self, cache_path: Optional[str] = None):
        self.cache = {}
        if cache_path is not None:
            self.load_cache(cache_path)
    
    def retrieve(self, query: Union[str, List[str]], topk: int = 10) -> List[List[Dict[str, Any]]]:
        raise NotImplementedError("Retrieve is an abstract method")

    def format_results(self, results: List[Dict[str, Any]]) -> str:
        raise NotImplementedError("Format results is an abstract method")

    def load_cache(self, cache_path: str):
        assert os.path.exists(cache_path), f"Cache file {cache_path} does not exist"
        with open(cache_path, "r") as f:
            self.cache = json.load(f)

    def save_cache(self, cache_path: str):
        with open(cache_path, "w") as f:
            json.dump(self.cache, f)


class DenseRetriever(Retriever):
    """
    In dense retriever, we rely on a dense index for efficient retrieval, and then use a corpus to map the ids back to the original data.
    Just avoiding importing classes unless needed...
    """
    def __init__(self,
        index= None, # Optional[DenseIndex]
        corpus = None, # Optional[Corpus]
        index_options: Optional[IndexOptions] = None,
        encoder_options: Optional[ModelOptions] = None,
        corpus_options: Optional[CorpusOptions] = None,
    ):
        """
        Initialize the DenseIndex. 
        We recommend using a pre-constructed index as opposed to initializing the index here.
        Args:
            index: Optional[DenseIndex]
                A pre-constructed index
            corpus: Optional[Corpus]
                A pre-constructed corpus
            index_options: Optional[IndexOptions]
                The options for the index
            encoder_options: Optional[ModelOptions]
                The options for the encoder
            corpus_options: Optional[CorpusOptions]
                The options for the corpus
        """
        from retrievaltools.encoder import load_encoder
        from retrievaltools.index import load_index
        from retrievaltools.data import load_corpus

        if index is None:
            assert encoder_options is not None and index_options is not None, "encoder_options and index_options must be provided if index is not provided"
            encoder = load_encoder(encoder_options)
            self.index = load_index(index_options, encoder)
        else:
            self.index = index

        # if corpus is not provided, we will not map the ids back to the original data 
        # (since the index may already contain all the information we need if the texts were saved)
        if corpus is None:
            if corpus_options is not None:
                self.corpus = load_corpus(corpus_options)
            else:
                self.corpus = None
        else:
            self.corpus = corpus


    def retrieve(self, query: Union[str, List[str]], topk: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Retrieve the topk results for the query.
        The key difference between this and index.get_topk is that this will return the results in a more user-friendly format. We need to map the ids back to the original data using a corpus.

        In the results, we guarantee "id", "score", and "text" are present. There may be other metadata fields depending on the corpus.
        """
        if isinstance(query, str):
            query = [query]

        results = self.index.get_topk(query, topk=topk)
        # map the ids back to the original data
        outputs = []
        
        for query, res in zip(query, results):
            ctxs = []
            for idx in range(len(res["id"])):
                ctx = {k: v[idx] for k, v in res.items()}
                ctx["position"] = idx + 1
                if "texts" in ctx and isinstance(ctx["texts"], dict):
                    ctx.update(ctx.pop("texts"))
                if self.corpus is not None:
                    ctx.update(self.corpus.get_item(res["id"][idx]))
                ctxs.append(ctx)
            outputs.append(ctxs)
        return outputs


    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format the results into a string.
        """
        template = "Result {position}\nTitle: {title}\nURL: {url}\n{long_snippet}"
        return "\n".join([template.format(**r) for r in results])


class WebSearchRetriever(Retriever):
    """
    In web search retriever, we rely on a web search engine to retrieve the topk results for the query.
    Right now, we only support Serper
    TODO: add more search engines
    """
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: str = "https://google.serper.dev/search", 
        min_delay: float = 0.001, 
        max_delay: float = 0.003,
        use_cache: bool = True,
        use_crawl4ai: bool = False,
    ):
        if api_key is None:
            self.api_key = os.environ.get("SERPER_API_KEY")
            assert self.api_key is not None, "SERPER_API_KEY must be set"
        else:
            self.api_key = api_key
        self.base_url = base_url
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            "X-API-KEY": self.api_key,
            'Content-Type': 'application/json'
        }
        self.use_cache = use_cache
        self.use_crawl4ai = use_crawl4ai
        if use_cache:
            self.CACHE_PATH = "cache/serper_search_cache.json"
            self.cache_file = ThreadSafeFileHandler(self.CACHE_PATH)
            self.cache = self.cache_file.read_data()
        
    def _add_delay(self):
        """Add a random delay between requests to appear more human-like"""
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)

    def retrieve(self, query: Union[str, List[str]], topk: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Retrieve the topk results for the query.
        
        In Serper, the results contain: "title", "url", "date", "text", "position"
        """
        results = []

        if isinstance(query, str):
            query = [query]

        for q in tqdm(query, desc="Retrieving from Serper"):
            if self.use_cache and q in self.cache:
                results.append(self.cache[q])
                continue
           
            payload = json.dumps({
                "q": q,
                "num": 10,
            })
            response = requests.post(url=self.base_url, data=payload, headers=self.headers)
            results.append(response.json())
            if self.use_cache:
                self.cache[q] = response.json()
        
        # update cache
        if self.use_cache:
            threading.Thread(target=lambda: self.cache_file.update_data(lambda x: x.update(self.cache) or x), daemon=True).start()

        # further processing of the results
        outputs = []
        results = [r for r in results if 'organic' in r]
        
        for result in tqdm(results, desc="Scraping results"):
            urls = [r['link'] for r in result['organic']]
            snippets = [r['snippet'] for r in result['organic']]
            if self.use_crawl4ai:
                scraped_results = asyncio.run(scrape_page_content_crawl4ai_batch(urls, snippets))
            else:
                scraped_results = [scrape_page_content(url, snippet=snippet, num_characters=2000) for url, snippet in zip(urls, snippets)]
            
            for r, (success, snippet, fulltext) in zip(result['organic'], scraped_results):
                r["text"] = r.pop("snippet")
                r["url"] = r.pop("link")

                if success:
                    r["long_snippet"] = snippet
                    r["full_text"] = fulltext

            outputs.append(result['organic'])

        return outputs

    def format_results(self, results: List[Dict[str, Any]], topk: int = 10) -> str:
        """
        Format the results into a string.
        """
        keys = ["title", "url", "long_snippet"]
        # template = "Result {position}\nTitle: {title}\nURL: {url}\n{long_snippet}"
        template = "<Search Result {position}>\n<Title: {title}>\n<URL: {url}>\n{long_snippet}\n</Search Result {position}>"
        # for some websites, we may not be able to scrape the page content
        results = [r for r in results if all(k in r for k in keys) and all(not isinstance(r[k], str) or len(r[k]) > 0 for k in keys)][:topk]
        results = [{"position": i+1, **r} for i, r in enumerate(results)]
        return "\n\n".join([template.format(**r) for r in results])


class EndPointRetriever(Retriever):
    """
    In end point retriever, we rely on a end point to retrieve the topk results for the query.
    """
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.end_point = f"http://{self.host}:{self.port}"

    def retrieve(self, query: Union[str, List[str]], topk: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Retrieve the topk results for the query.
        """
        if isinstance(query, str):
            query = [query]
        payload = {"query": query, "topk": topk}
        response = requests.post(self.end_point + "/retrieve_batch/", params=payload)
        return response.json()

    def format_results(self, results: List[Dict[str, Any]], topk: int = 10) -> str:
        """
        Format the results into a string.
        """
        keys = ["position", "title", "url", "long_snippet"]
        template = "Result {position}\nTitle: {title}\nURL: {url}\n{long_snippet}"
        results = [r for r in results if all(k in r for k in keys) and all(r[k] != "" for k in keys)][:topk]
        return "\n\n".join([template.format(**r) for r in results])


def load_retriever(
    retriever_options: RetrieverOptions, 
    index_options: Optional[IndexOptions] = None, 
    model_options: Optional[ModelOptions] = None, 
    corpus_options: Optional[CorpusOptions] = None
):
    if retriever_options.retriever_type == "dense":
        return DenseRetriever(
            index_options=index_options,
            encoder_options=model_options,
            corpus_options=corpus_options if retriever_options.include_corpus else None,
        )
    elif retriever_options.retriever_type == "web":
        return WebSearchRetriever(
            api_key=retriever_options.api_key,
            use_cache=retriever_options.use_cache,
            use_crawl4ai=retriever_options.use_crawl4ai,
        )
    elif retriever_options.retriever_type == "endpoint":
        return EndPointRetriever(retriever_options.host, retriever_options.port)
    else:
        raise ValueError(f"Invalid retriever type: {retriever_options.retriever_type}")
