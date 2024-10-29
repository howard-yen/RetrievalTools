
from index import Indexer

import pickle
import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Union, Tuple
from utils import mean_pooling, max_pooling, cls_pooling
from tqdm import trange

import logging
logger = logging.getLogger(__name__)


class STRetriever():
    # sentence transformer retriever, preferred due to multi-gpu support
    # recommend using torch dtype = float16
    def __init__(self, 
        model_name_or_path: str, 
        ctx_size: Optional[int] = None, 
        batch_size: int = 32, 
        dtype: str = "float16",
        **model_kwargs,
    ):
        torch_dtype = dtype if dtype in ["auto", None] else getattr(torch, dtype)
        self.model = SentenceTransformer(
            model_name_or_path, 
            device="cuda", 
            trust_remote_code=True,
            model_kwargs={"trust_remote_code": True, "torch_dtype": torch_dtype, **model_kwargs},
            tokenizer_kwargs={"trust_remote_code": True}
        ).eval()
        self.tokenizer = self.model.tokenizer
        
        if ctx_size is not None:
            self.model.max_seq_length = ctx_size
    
            if ctx_size > self.model.max_seq_length:
                logger.warning(f"ctx_size {ctx_size} is greater than model max_seq_length {self.model.max_seq_length}, this may return unexpected results")

        self.config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.hidden_size = self.config.hidden_size
        self.dtype = dtype
        self.normalize_embedding = True
        self.encode_batch_size = batch_size
        self.index_batch_size = 2048
        # add quantization support here
        self.index = Indexer(dtype=dtype, vector_sz=self.hidden_size)

    @torch.no_grad()
    def encode(self, context: List[str], prompt: str = "") -> np.array:
        """
        Returns an np.array
        """
        if len(context) > 1000 and torch.cuda.device_count() > 1:
            pool = self.model.start_multi_process_pool()
            emb = self.model.encode_multi_process(
                context, 
                pool, 
                normalize_embeddings=self.normalize_embedding, 
                prompt=prompt, 
                batch_size=self.encode_batch_size
            )
            self.model.stop_multi_process_pool(pool)
        else:
            emb = self.model.encode(
                context, 
                normalize_embeddings=self.normalize_embedding, 
                prompt=prompt, 
                show_progress_bar=True, 
                convert_to_numpy=True, 
                batch_size=self.encode_batch_size
            )
        return emb


    def build_index(
        self, 
        data: Optional[List[str]] = None, 
        ids: Optional[List[str]] = None,
        emb_files: Optional[Union[List[str], str]] = None, 
        prompt: str = "",
    ) -> None:
        """
        Build the index, either from the data or from the embeddings files
        """
        if emb_files is not None:
            logger.info("Building index from embeddings files")
            # self.index = Indexer(vector_sz=self.hidden_size) # do we want to reset the index here? 
            if isinstance(emb_files, str):
                emb_files = [emb_files]
            for file in emb_files:
                with open(file, "rb") as f:
                    ids, emb = pickle.load(f)
                    self.index.index_data(ids, emb)
        else:
            assert data is not None and ids is not None
            logger.info("Building index from data")
            emb = self.encode(data, prompt=prompt)
            self.index.index_data(ids, emb)
        logger.info("Index built, moving to GPU")
        self.index.to_gpu()
    
    
    @property
    def index_size(self) -> int:
        return self.index.index.ntotal


    def reset_index(self) -> None:
        """
        Reset the index of the retriever
        """
        del self.index
        self.index = Indexer(vector_sz=self.hidden_size)
    

    def get_topk(
        self, 
        queries: Optional[Union[List[str], str]] = None, 
        query_prompt:str = "", 
        query_emb: Optional[np.array] = None,
        topk: int = 10,
    ) -> List[Tuple[List[object], np.array]]:
        """
        Get the tok k results for the queries. If query_emb is provided, queries must be None
        
        Returns a list of tuples containing the external ids and scores of the top documents
        """

        if query_emb is not None:
            assert queries is None, "queries must be None if query_emb is provided"
            q_emb = query_emb
        else:
            if isinstance(queries, str):
                queries = [queries]
            q_emb = self.encode(queries, prompt=query_prompt)

        results = self.index.search_knn(q_emb, top_docs=topk, index_batch_size=self.index_batch_size)
        return results


class HFRetriever():
    def __init__(self, 
        model_name_or_path: str, 
        ctx_size: Optional[int] = None, 
        batch_size: int = 32, 
        dtype: str = "float16",
        pooling: str = "mean", # other options: "cls", "max"
        **kwargs
    ):
        assert pooling in ["mean", "cls", "max"], "Pooling must be one of 'mean', 'cls', 'max'"
    
        torch_dtype = dtype if dtype in ["auto", None] else getattr(torch, dtype)
        self.config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, 
            # device_map="auto", 
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **kwargs,
        ).eval().to("cuda")
        
        self.ctx_size = ctx_size
        if ctx_size is None:
            self.ctx_size = self.config.max_position_embeddings
        elif ctx_size > self.config.max_position_embeddings:
            logger.warning(f"ctx_size {ctx_size} is greater than model max_seq_length {self.model.max_position_embeddings}, this may return unexpected results")

        self.hidden_size = self.config.hidden_size
        self.dtype = dtype
        self.batch_size = batch_size
        self.pooling = pooling
        self.index = Indexer(dtype=dtype, vector_sz=self.hidden_size)


    @torch.no_grad()
    def encode(self, context: List[str], prompt: str = "") -> np.array:
        """
        Returns an np.array
        """
        allemb = []
        # TODO: for particuarly large datasets, we can wrap everything around a dataloader to save time on the tokenization
        for i in trange(0, len(context), self.batch_size):
            batch = context[i:i+self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.ctx_size)
            outputs = self.model(**inputs.to(self.model.device))

            if self.pooling == "mean":
                embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            elif self.pooling == "cls":
                embeddings = cls_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            elif self.pooling == "max":
                embeddings = max_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            allemb.append(embeddings.cpu().numpy())
        return np.cat(allemb, axis=0)


    def build_index(
        self, 
        data: Optional[List[str]] = None, 
        ids: Optional[List[str]] = None,
        emb_files: Optional[List[str]] = None, 
        prompt: str = "",
    ):
        """
        Build the index, either from the data or from the embeddings files
        """
        if emb_files is not None:
            logger.info("Building index from embeddings files")
            self.index = Indexer()
            for file in emb_files:
                with open(file, "rb") as f:
                    ids, emb = pickle.load(f)
                    self.index.index_data(ids, emb)
        else:
            assert data is not None and ids is not None
            logger.info("Building index from data")
            emb = self.encode(data, prompt=prompt)
            self.index.index_data(ids, emb)
        logger.info("Index built, moving to GPU")
        self.index.to_gpu()


    def reset_index(self):
        self.index = Indexer()
    

    def get_topk(
        self, 
        queries: Union[List[str], str], 
        query_prompt:str = "",
        topk: int = 10,
    ):
        if isinstance(queries, str):
            queries = [queries]
        q_emb = self.encode(queries, prompt=query_prompt)
        results = self.index.search_knn(q_emb, top_docs=topk)
        return results


def load_retriever(
    model_name_or_path: str, 
    max_length: int = 512,
    batch_size: int = 32,
    use_hf: bool = False,
):
    if use_hf:
        kwargs = {}
        if "gte" in model_name_or_path:
            kwargs = {"unpad_inputs": True, "use_memory_efficient_attention": True, "pooling": "cls"}
        return HFRetriever(model_name_or_path, ctx_size=max_length, batch_size=batch_size, **kwargs)
    return STRetriever(model_name_or_path, ctx_size=max_length, batch_size=batch_size)
