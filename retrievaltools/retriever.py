import os
import json
import random
import time
import requests
import asyncio

from typing import Optional, List, Dict, Any, Union
from functools import lru_cache
from dataclasses import dataclass
from tqdm import tqdm
from diskcache import Cache

from retrievaltools.arguments import CorpusOptions, IndexOptions, ModelOptions, RetrieverOptions
from retrievaltools.utils import init_logger, serper_search, scrape_pdf, scrape_html, detect_content_type

logger = init_logger(__name__)
cache = Cache(os.environ.get("RT_CACHE_PATH", "./cache"))

@dataclass
class QueryResult:
    # required for all retrievers
    id: str
    position: int
    
    # optional
    title: Optional[str] = None
    text: Optional[str] = None
    score: Optional[float] = None


@dataclass
class RetrieveResult:
    query: str
    results: List[QueryResult]
    formatted_output: str


@dataclass
class VisitResult:
    id: str
    content: str


class Retriever():
    """
    The retriever class is a simple interface that takes queries as inputs and return results. 
    The retriever also supports visiting a specific document given its id (e.g., URL for web search retriever).
    """
    def __init__(self):
        pass
    
    def retrieve(self, query: Union[str, List[str]], topk: int = 10) -> List[RetrieveResult]:
        """
        Given queries, return the topk results. 
        
        The output is a list of RetrieveResult objects. Each RetrieveResult object contains the following:
         - query: str = the query
         - results: List[QueryResult] = List of QueryResult objects, note that some content may be retriever-dependent
         - formatted_output: str = A formatted string of the results, which is also retriever-specific
        """
        raise NotImplementedError("Retrieve is an abstract method")

    def visit(self, id: str) -> VisitResult:
        """
        Fetch the content of a document given its id.

        The output contains the id and the content of the document.
        """
        raise NotImplementedError("Visit is an abstract method")


class DenseRetriever(Retriever):
    """
    In dense retriever, we rely on a dense index for efficient retrieval, and then use a corpus to map the ids back to the original data.
    Just avoiding importing classes unless needed...
    """
    def __init__(self,
        index = None, # Optional[DenseIndex]
        corpus = None, # Optional[Corpus]
        index_options: Optional[IndexOptions] = None,
        encoder_options: Optional[ModelOptions] = None,
        corpus_options: Optional[CorpusOptions] = None,
        snippet_length: int | None = None,
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
            snippet_length: int
                The length of the snippet to return in formatted search results. Return all text if None
        """

        if index is None:
            assert encoder_options is not None and index_options is not None, "encoder_options and index_options must be provided if index is not provided"
            from retrievaltools.encoder import load_encoder
            from retrievaltools.index import load_index
            encoder = load_encoder(encoder_options)
            self.index = load_index(index_options, encoder)
        else:
            self.index = index

        # if corpus is not provided, we will not map the ids back to the original data 
        # (since the index may already contain all the information we need if the texts were saved)
        self.corpus = None
        if corpus is not None:
            self.corpus = corpus
        elif corpus_options is not None:
            from retrievaltools.data import load_corpus
            self.corpus = load_corpus(corpus_options)

        self.snippet_length = snippet_length


    # HY: I don't think we should do the diskcache here, because the function depends on the model and corpus
    # @lru_cache(maxsize=8192)
    def retrieve(self, query: Union[str, List[str]], topk: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Retrieve the topk results for the query.
        Each query results will always have "id", "position", and "score".
        If the corpus is provided, we will also have "text" and "title" (if available in the corpus).
        """
        if isinstance(query, str):
            query = [query]

        results = self.index.get_topk(query, topk=topk)
        outputs = []
        
        for query, res in zip(query, results):
            ctxs = []
            for idx in range(len(res["id"])):
                ctx = {k: v[idx] for k, v in res.items()}
                ctx["position"] = idx + 1
                # HY: not sure why texts can be dict... should probably fix later
                if "texts" in ctx and isinstance(ctx["texts"], dict):
                    ctx.update(ctx.pop("texts"))
                # if corpus is provided, get additional metadata from the corpus
                if self.corpus is not None:
                    ctx.update(self.corpus.get_item(res["id"][idx]))
                ctxs.append(ctx)
            ctxs = [QueryResult(ctx["id"], ctx["position"], ctx.get("title", None), ctx.get("text", None), ctx.get("score", None)) for ctx in ctxs]
            outputs.append(RetrieveResult(query, ctxs, self.format_results(query, ctxs)))
        
        return outputs


    @lru_cache(maxsize=8192)
    def visit(self, id: str, query: str, content_length: int = 10000, scoring_func: str = "rouge", chunking_func: str = "newline") -> str:
        """
        Fetch the content of a document given its id. 
        Returns the text of the document, with the query and the content length used to find the most relevant snippet.
        """
        if self.corpus is None:
            raise ValueError("Corpus is not provided. Please provide a corpus when initializing the retriever if you want to use the visit method.")
        text = self.corpus.get_text(id)

        content = text[:content_length]
        if query is not None:
            content = find_snippet(text, query, content_length, scoring_func, chunking_func)
        return content


    def format_results(self, query: str, results: List[QueryResult]) -> str:
        """
        Format the results into a string.
        No need to cache.
        """
        # s = f"Query: {query}\nThe retriever returned {len(results)} results:\n\n"
        s = f"The retriever returned {len(results)} results:\n\n"

        for res in results:
            s += f"<Result {res.position}>\n<Document ID>{res.id}</Document ID>\n"
            if res.title is not None:
                s += f"<Title>{res.title}</Title>\n"
            if res.text is not None:
                if self.snippet_length is not None:
                    s += f"<Text>\n{res.text[:self.snippet_length]}...\n</Text>\n"
                else:
                    s += f"<Text>\n{res.text}\n</Text>\n"
            s += f"</Result {res.position}>\n\n"

        return s


class WebSearchRetriever(Retriever):
    """
    In web search retriever, we rely on a web search engine to retrieve the topk results for the query.
    Right now, we only support Serper
    TODO: add more search engines
    """
    def __init__(
        self, 
        web_scraping: str = "none",
    ):
        self.web_scraping = web_scraping
        

    def _add_delay(self):
        """Add a random delay between requests to appear more human-like"""
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)


    def retrieve(self, query: Union[str, List[str]], topk: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Retrieve the topk results for the query.
        We only return the snippet here, for full text, use the visit method.
        
        In Serper, the results contain: "title", "url", "date", "text", "position"
        """
        results = []

        if isinstance(query, str):
            query = [query]

        for q in tqdm(query, desc="Retrieving from Serper", disable=not self.verbose):
            response = serper_search(q, topk=topk)
            results.append(response['organic'])

        # further processing of the results
        outputs = []

        for query, result in zip(query, results):
            result = [QueryResult(
                id=r['link'],
                position=i+1,
                title=r.get('title', None),
                text=r.get('snippet', None),
                score=r.get('score', None),
            ) for i, r in enumerate(result)]
            outputs.append(RetrieveResult(query, result, self.format_results(query, result)))
       
        return outputs
    

    def visit(self, url: str, query: str, content_length: int = 10000, scoring_func: str = "rouge", chunking_func: str = "newline") -> str:
        """
        Fetch the content of a document given its url.
        Returns the text of the document, with the query and the content length used to find the most relevant snippet.
        Returns None if the content fails to be extracted.
        """
        try:
            content_type = detect_content_type(url)
            if content_type == "pdf":
                result = asyncio.run(scrape_pdf(url))
            else:
                result = asyncio.run(scrape_html(url))
            return result
        except Exception as e:
            return None


    def format_results(self, results: List[Dict[str, Any]], topk: int = 10) -> str:
        """
        Format the results into a string.
        """
        # s = f"Query: {query}\nThe search engine returned {len(results)} results:\n\n"
        s = f"The search engine returned {len(results)} results:\n\n"

        for res in results:
            s += f"<Result {res.position}>\n<URL>{res.id}</URL>\n"
            if res.title is not None:
                s += f"<Title>{res.title}</Title>\n"
            if res.text is not None:
                s += f"<Snippet>\n{res.text}\n</Snippet>\n"
            s += f"</Result {res.position}>\n\n"
        return s


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
            web_scraping=retriever_options.web_scraping,
            verbose=retriever_options.verbose,
        )
    elif retriever_options.retriever_type == "endpoint":
        return EndPointRetriever(retriever_options.host, retriever_options.port)
    else:
        raise ValueError(f"Invalid retriever type: {retriever_options.retriever_type}")
