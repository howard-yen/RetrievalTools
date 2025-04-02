import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import http.client
from dataclasses import asdict

from tqdm import trange, tqdm
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding
from sentence_transformers import SentenceTransformer

from encoder import Encoder, load_encoder
from index import DenseIndex
from data import Corpus
from arguments import CorpusOptions, IndexOptions, ModelOptions

import logging
logger = logging.getLogger(__name__)

class Retriever():
    """
    The retriever class is a simple interface that takes queries as inputs and return results. 
    """
    def __init__(self):
        raise NotImplementedError("Retriever is an abstract class")
    
    def retrieve(self, queries: List[str], topk: int = 10) -> List[List[Dict[str, Any]]]:
        raise NotImplementedError("Retrieve is an abstract method")


class DenseRetriever(Retriever):
    """
    In dense retriever, we rely on a dense index for efficient retrieval, and then use a corpus to map the ids back to the original data.
    """
    def __init__(self,
        index: Optional[DenseIndex] = None,
        corpus: Optional[Corpus] = None,
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
            encoder_options: Optional[EncoderOptions]
                The options for the encoder
            corpus_options: Optional[CorpusOptions]
                The options for the corpus
        """
        if index is None:
            assert encoder_options is not None and index_options is not None, "encoder_options and index_options must be provided if index is not provided"
            encoder = Encoder(
                model_name_or_path=encoder_options.model_name_or_path,
                max_length=encoder_options.max_length,
                batch_size=encoder_options.batch_size,
                use_hf=encoder_options.use_hf,
            )
            index = DenseIndex(**asdict(index_options), encoder=encoder)
        else:
            self.index = index

        if corpus is None:
            assert corpus_options is not None, "corpus_options must be provided if corpus is not provided"
            self.corpus = Corpus(**asdict(corpus_options))
        else:
            self.corpus = corpus

    def retrieve(self, queries: List[str], topk: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Retrieve the topk results for the query.
        The key difference between this and index.get_topk is that this will return the results in a more user-friendly format. We need to map the ids back to the original data using a corpus.
        """
        results = self.index.get_topk(queries, topk=topk)
        # map the ids back to the original data
        outputs = []
        
        for query, (ids, scores) in zip(queries, results):
            ctxs = []
            for id, score in zip(ids, scores):
                ctxs.append({
                    "id": id,
                    "score": score,
                    "text": self.corpus.get_text(id)
                })
            outputs.append(ctxs)
        return outputs


class WebSearchRetriever(Retriever):
    """
    In web search retriever, we rely on a web search engine to retrieve the topk results for the query.
    """
    def __init__(self, api_key: str, base_url: str = "google.serper.dev"):
        self.api_key = api_key
        self.base_url = base_url
        self.conn = http.client.HTTPSConnection(self.base_url)

    def retrieve(self, queries: List[str], topk: int = 10) -> List[List[Dict[str, Any]]]:
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
        
        # further processing of the results
        outputs = []
        for result in results:
            # in Serper, organic contains "title", "link", "date", "snippet", "position"
            output.append(result['organic'])

        return results