from typing import List, Dict, Any

from simple_parsing import ArgumentParser
import uvicorn
from fastapi import FastAPI, Query

from retrievaltools.arguments import RetrieverOptions, ModelOptions, CorpusOptions, IndexOptions
from retrievaltools.retriever import DenseRetriever, WebSearchRetriever
from retrievaltools.utils import init_logger

logger = init_logger(__name__)


def test_retriever():
    import requests
    payload = {"query": "Who are the current MVPs in the MLB?"}
    response = requests.post("http://localhost:8000/retrieve", params=payload)
    # response = requests.post("http://0.0.0.1:8001/retrieve", params=payload)
    print(response.json())

    payload = {"query": ["Who are the current MVPs in the MLB?", "What is the most popular Taiwanese film of all time?"]}
    response = requests.post("http://localhost:8000/retrieve_batch", params=payload)
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

    @app.post("/retrieve")
    @app.post("/search")
    def retrieve(query: str, topk: int = 10) -> Dict[str, Any]:
        output = retriever.retrieve([query], topk=topk)[0]
        return {"output": output}

    @app.post("/retrieve_batch")
    @app.post("/search_batch")
    def retrieve_batch(query: List[str] = Query(...), topk: int = 10) -> Dict[str, Any]:
        output = retriever.retrieve(query, topk=topk)
        return {"output": output}

    @app.post("/visit")
    @app.post("/open_url")
    def visit(id: str, query: str = None, content_length: int = 10000, scoring_func: str = "rouge", chunking_func: str = "newline") -> str:
        return {"output": retriever.visit(id, query, content_length=content_length, scoring_func=scoring_func, chunking_func=chunking_func)}

    @app.post("/ping")
    def ping():
        return {"status": "ok"}

    uvicorn.run(app, host=args.retriever_options.host, port=args.retriever_options.port)


if __name__ == "__main__":
    main()
