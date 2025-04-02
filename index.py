# Adapted from https://github.com/facebookresearch/contriever/blob/main/src/index.py
import os
import pickle
from typing import List, Tuple, Optional, Union, Dict, Any

import faiss
import torch
import numpy as np
from tqdm import tqdm
from utils import init_logger

logger = init_logger(__name__)

class Indexer(object):
    """
    Adopted from Contriever, but we also support storing the texts
    """

    def __init__(
        self, 
        vector_sz: int, 
        n_subquantizers: int = 0, 
        n_bits: int = 8, 
        dtype: str = "float16"
    ):
        """
        Intialize the indexer
        """
        if n_subquantizers > 0:
            self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(vector_sz)
        self.index_id_to_db_id = []
        self.index_id_to_text = []
        self.dtype = dtype

    def to_gpu(self):
        """
        Move the index to multiple GPUs if available
        """
        logger.info(f"Moving index to gpu, found {faiss.get_num_gpus()} GPUs")
        resources = [faiss.StandardGpuResources() for i in range(faiss.get_num_gpus())]
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True if self.dtype == "float16" else False
        co.shard = True
        index_gpu = faiss.index_cpu_to_gpu_multiple_py(resources, self.index, co=co)
        self.index = index_gpu

    def index_data(self, ids: List[str], embeddings: np.array, texts: List[str] = None):
        """
        Index the data
        """
        if texts is not None:
            assert self._text_consistent(), "Text mapping is not consistent, cannot add text unless already consistent"
            self._update_text_mapping(texts)
        self._update_id_mapping(ids)

        embeddings = embeddings.astype(self.dtype)
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)

        logger.info(f'Total data indexed {len(self.index_id_to_db_id)}')

    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size: int = 2048) -> List[Dict[str, Any]]:
        """
        Returns a list of tuples containing the external ids and scores of the top documents (np array of float32)
        """
        query_vectors = query_vectors.astype(self.dtype)
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        for k in tqdm(range(nbatch), desc="searching knn", disable=nbatch<5):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
            if self._text_consistent():
                db_texts = [[self.index_id_to_text[i] for i in query_top_idxs] for query_top_idxs in indexes]
                result.extend([{"id": db_ids[i], "score": scores[i], "text": db_texts[i]} for i in range(len(db_ids))])
            else:
                result.extend([{"id": db_ids[i], "score": scores[i]} for i in range(len(db_ids))])

        logger.info("Finished searching knn")
        return result

    def serialize(self, dir_path: str):
        """
        Serialize the index (save to dir_path)
        """
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        logger.info(f'Serializing index to {index_file}, meta data to {meta_file}')

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

        if self._text_consistent():
            with open(os.path.join(dir_path, 'index_text.faiss'), mode='wb') as f:
                pickle.dump(self.index_id_to_text, f)


    def deserialize_from(self, dir_path: str):
        """
        Deserialize the index (load from dir_path)
        """
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        logger.info(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)
        logger.info('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

        if os.path.exists(os.path.join(dir_path, 'index_text.faiss')):  
            with open(os.path.join(dir_path, 'index_text.faiss'), "rb") as reader:
                self.index_id_to_text = pickle.load(reader)


    def _update_id_mapping(self, db_ids: List[str]):
        """
        Update the id mapping
        """
        self.index_id_to_db_id.extend(db_ids)

    def _update_text_mapping(self, texts: List[str]):
        """
        Update the text mapping
        """
        self.index_id_to_text.extend(texts)

    def _text_consistent(self) -> bool:
        """
        Check if the text mapping is consistent with the index mapping
        """
        return len(self.index_id_to_text) == len(self.index_id_to_db_id)


class DenseIndex():
    """
    This is essentially a wrapper around the Indexer class with the encoding functionality built in
    """
    def __init__(
        self, 
        encoder, 
        dtype: str = "float16",
        index_batch_size: int = 2048,
        n_subquantizers: int = 0,
        n_bits: int = 8,
    ):
        """
        Initialize the DenseIndex
        Args:
            encoder: encoder.Encoder
            dtype: str
            index_batch_size: int
            n_subquantizers: int
            n_bits: int
        """
        self.encoder = encoder 
        self.hidden_size = self.encoder.hidden_size
        self.n_subquantizers = n_subquantizers
        self.n_bits = n_bits

        self.torch_dtype = dtype if dtype in ["auto", None] else getattr(torch, dtype)
        self.dtype = dtype
        self.index_batch_size = index_batch_size

        # add quantization support here
        self.index = Indexer(
            dtype=dtype, 
            vector_sz=self.hidden_size, 
            n_subquantizers=n_subquantizers,
            n_bits=n_bits,
        )


    def build_index(
        self, 
        data: Optional[List[str]] = None, 
        ids: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
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
                    data = pickle.load(f)
                    texts = None
                    if len(data) == 2:
                        ids, emb = data
                    elif len(data) == 3:
                        ids, emb, texts = data

                    self.index.index_data(ids, emb, texts)
        else:
            assert data is not None and ids is not None
            logger.info("Building index from data")
            emb = self.encoder.encode(data, prompt=prompt)
            self.index.index_data(ids, emb, texts)
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
        self.index = Indexer(
            dtype=self.dtype, 
            vector_sz=self.hidden_size, 
            n_subquantizers=self.n_subquantizers,
            n_bits=self.n_bits,
        )
    

    def get_topk(
        self, 
        queries: Optional[Union[List[str], str]] = None, 
        query_prompt: str = "", 
        query_emb: Optional[np.array] = None,
        topk: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get the tok k results for the queries. If query_emb is provided, queries must be None
        
        Returns a list of dictionaries containing the external ids, scores, and texts of the top documents
        """

        if query_emb is not None:
            assert queries is None, "queries must be None if query_emb is provided"
            q_emb = query_emb
        else:
            if isinstance(queries, str):
                queries = [queries]
            q_emb = self.encoder.encode(queries, prompt=query_prompt)

        results = self.index.search_knn(q_emb, top_docs=topk, index_batch_size=self.index_batch_size)
        return results
