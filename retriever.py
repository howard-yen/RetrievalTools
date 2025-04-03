import os
import json
from typing import Optional, List, Dict, Any
import http.client
from dataclasses import asdict

from arguments import CorpusOptions, IndexOptions, ModelOptions

import logging
logger = logging.getLogger(__name__)

class Retriever():
    """
    The retriever class is a simple interface that takes queries as inputs and return results. 

    In the results, we guarantee that "text" is present. Other fields are dependent on the specific retriever (see subclasses).
    """
    def __init__(self):
        raise NotImplementedError("Retriever is an abstract class")
    
    def retrieve(self, queries: List[str], topk: int = 10) -> List[List[Dict[str, Any]]]:
        raise NotImplementedError("Retrieve is an abstract method")


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

    def retrieve(self, queries: List[str], topk: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Retrieve the topk results for the query.
        The key difference between this and index.get_topk is that this will return the results in a more user-friendly format. We need to map the ids back to the original data using a corpus.

        In the results, we guarantee "id", "score", and "text" are present. There may be other metadata fields depending on the corpus.
        """
        results = self.index.get_topk(queries, topk=topk)
        # map the ids back to the original data
        outputs = []
        
        for query, res in zip(queries, results):
            ctxs = []
            for idx in range(len(res["id"])):
                ctx = {k: v[idx] for k, v in res.items()}
                if "texts" in ctx and isinstance(ctx["texts"], dict):
                    ctx.update(ctx.pop("texts"))
                if self.corpus is not None:
                    ctx.update(self.corpus.get_item(res["id"][idx]))
                ctxs.append(ctx)
            outputs.append(ctxs)
        return outputs


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

    def retrieve(self, queries: List[str], topk: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Retrieve the topk results for the query.
        
        In Serper, the results contain: "title", "url", "date", "text", "position"
        """
        results = []

        for query in queries:
            payload = json.dumps({
                "q": query
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
                r["text"] = r.pop("snippet")
                r["url"] = r.pop("link")
            outputs.append(result['organic'])

        return outputs
