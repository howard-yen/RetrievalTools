import os
import json
import string
from typing import Optional, List, Dict, Any, Union, Tuple
import http.client
from dataclasses import asdict
import requests

from retrievaltools.arguments import CorpusOptions, IndexOptions, ModelOptions, RetrieverOptions
from retrievaltools.utils import scrape_page_content, init_logger

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
        from encoder import load_encoder
        from index import load_index
        from data import load_corpus

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
    def __init__(self, api_key: str = None, base_url: str = "google.serper.dev"):
        if api_key is None:
            self.api_key = os.environ.get("SERPER_API_KEY")
            assert self.api_key is not None, "SERPER_API_KEY must be set"
        else:
            self.api_key = api_key
        self.base_url = base_url
        self.conn = http.client.HTTPSConnection(self.base_url)

    def retrieve(self, query: Union[str, List[str]], topk: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Retrieve the topk results for the query.
        
        In Serper, the results contain: "title", "url", "date", "text", "position"
        """
        results = []

        if isinstance(query, str):
            query = [query]

        for q in query:
            payload = json.dumps({
                "q": q
            })
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            self.conn.request("POST", "/search", payload, headers)
            res = self.conn.getresponse()
            data = res.read()
            results.append(json.loads(data.decode("utf-8")))
        
        # further processing of the result
        outputs = []
        for result in results:
            # in Serper, organic contains "title", "link", "date", "snippet", "position"
            # rename snippet to text and link to url
            for r in result['organic']:
                if "snippet" not in r:
                    continue
                r["text"] = r.pop("snippet")
                r["url"] = r.pop("link")
                # scrape the page content, 2000 characters is ~250 tokens? maybe more
                success, snippet, fulltext = scrape_page_content(r["url"], snippet=r["text"], num_characters=2000)
                if success:
                    r["long_snippet"] = snippet
                    r["full_text"] = fulltext
            outputs.append(result['organic'])

        return outputs

    def format_results(self, results: List[Dict[str, Any]], topk: int = 10) -> str:
        """
        Format the results into a string.
        """
        keys = ["position", "title", "url", "long_snippet"]
        template = "Result {position}\nTitle: {title}\nURL: {url}\n{long_snippet}"
        # for some websites, we may not be able to scrape the page content
        results = [r for r in results if all(k in r for k in keys)][:topk]
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
        return WebSearchRetriever()
    else:
        raise ValueError(f"Invalid retriever type: {retriever_options.retriever_type}")