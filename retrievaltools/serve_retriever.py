from typing import List, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import time

from simple_parsing import ArgumentParser
import uvicorn
from fastapi import FastAPI, Query
import transformers

from retrievaltools.arguments import RetrieverOptions, ModelOptions, CorpusOptions, IndexOptions
from retrievaltools.retriever import DenseRetriever, WebSearchRetriever
from retrievaltools.utils import init_logger

# Suppress tokenizer warnings (e.g., Qwen2TokenizerFast __call__ vs pad method)
transformers.logging.set_verbosity_error()

logger = init_logger(__name__)


@dataclass
class RequestStats:
    """Track request statistics for monitoring server load."""
    active_requests: int = 0
    total_requests: int = 0
    # Breakdown by endpoint type
    active_search: int = 0
    total_search: int = 0
    active_visit: int = 0
    total_visit: int = 0
    # Items processed
    total_queries: int = 0
    total_doc_ids: int = 0
    # Latency tracking (in seconds)
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))  # Keep last 100
    max_latency: float = 0.0
    min_latency: float = float('inf')
    total_latency: float = 0.0

    def start_request(self, request_type: str) -> float:
        """Returns start time for latency tracking."""
        self.active_requests += 1
        self.total_requests += 1
        if request_type == "search":
            self.active_search += 1
            self.total_search += 1
        elif request_type == "visit":
            self.active_visit += 1
            self.total_visit += 1
        return time.time()

    def end_request(self, request_type: str, start_time: float, items_processed: int = 1):
        """Records latency and items processed."""
        latency = time.time() - start_time
        self.active_requests -= 1
        if request_type == "search":
            self.active_search -= 1
            self.total_queries += items_processed
        elif request_type == "visit":
            self.active_visit -= 1
            self.total_doc_ids += items_processed
        
        # Update latency stats
        self.latencies.append(latency)
        self.total_latency += latency
        self.max_latency = max(self.max_latency, latency)
        if latency < self.min_latency:
            self.min_latency = latency

    @property
    def avg_latency(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests

    @property
    def recent_avg_latency(self) -> float:
        if len(self.latencies) == 0:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "search": {
                "active": self.active_search,
                "total": self.total_search,
                "queries_processed": self.total_queries,
            },
            "visit": {
                "active": self.active_visit,
                "total": self.total_visit,
                "docs_processed": self.total_doc_ids,
            },
            "latency": {
                "avg_ms": round(self.avg_latency * 1000, 2),
                "recent_avg_ms": round(self.recent_avg_latency * 1000, 2),
                "max_ms": round(self.max_latency * 1000, 2),
                "min_ms": round(self.min_latency * 1000, 2) if self.min_latency != float('inf') else 0.0,
            },
        }


def test_retriever():
    import requests
    payload = {"query": "Who are the current MVPs in the MLB?"}
    response = requests.post("http://localhost:8000/retrieve", params=payload)
    # response = requests.post("http://0.0.0.1:8001/retrieve", params=payload)
    print(response.json())

    payload = {"query": ["Who are the current MVPs in the MLB?", "What is the most popular Taiwanese film of all time?"]}
    response = requests.post("http://localhost:8000/retrieve_batch", params=payload)
    print(response.json())

    payload = {"id": "5"}
    response = requests.post("http://localhost:8000/visit", params=payload)
    print(response.json())

    response = requests.get("http://localhost:8000/stats")
    print(response.json())


def main():
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(RetrieverOptions, dest="retriever_options")
    parser.add_arguments(ModelOptions, dest="model_options")
    parser.add_arguments(CorpusOptions, dest="corpus_options")
    parser.add_arguments(IndexOptions, dest="index_options")
    args = parser.parse_args()

    if args.retriever_options.retriever_type == "dense":
        retriever = DenseRetriever(
            index_options=args.index_options,
            encoder_options=args.model_options,
            corpus_options=args.corpus_options if args.retriever_options.include_corpus else None,
            snippet_length=args.retriever_options.snippet_length,
        )
    elif args.retriever_options.retriever_type == "web_search":
        retriever = WebSearchRetriever()

    app = FastAPI()
    stats = RequestStats()

    @app.post("/retrieve")
    @app.post("/search")
    def retrieve(query: str, topk: int = 10) -> Dict[str, Any]:
        start_time = stats.start_request("search")
        try:
            output = retriever.retrieve([query], topk=topk)[0]
            return {"output": output}
        finally:
            stats.end_request("search", start_time, items_processed=1)

    @app.post("/retrieve_batch")
    @app.post("/search_batch")
    def retrieve_batch(query: List[str] = Query(...), topk: int = 10) -> Dict[str, Any]:
        start_time = stats.start_request("search")
        try:
            output = retriever.retrieve(query, topk=topk)
            return {"output": output}
        finally:
            stats.end_request("search", start_time, items_processed=len(query))

    @app.post("/visit")
    @app.post("/open_url")
    def visit(id: str, query: str = None, content_length: int = 10000, scoring_func: str = "rouge", chunking_func: str = "newline") -> Dict[str, str]:
        start_time = stats.start_request("visit")
        try:
            return {"output": retriever.visit(id, query, content_length=content_length, scoring_func=scoring_func, chunking_func=chunking_func)}
        finally:
            stats.end_request("visit", start_time, items_processed=1)

    @app.get("/stats")
    def get_stats() -> Dict[str, Any]:
        return stats.to_dict()

    @app.post("/ping")
    def ping():
        return {"status": "ok"}

    uvicorn.run(app, host=args.retriever_options.host, port=args.retriever_options.port)


if __name__ == "__main__":
    main()
