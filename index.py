# Adapted from https://github.com/facebookresearch/contriever/blob/main/src/index.py
import os
import pickle
from typing import List, Tuple
import logging

import faiss
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Indexer(object):

    def __init__(self, vector_sz: int, n_subquantizers:int = 0, n_bits: int = 8, dtype: str = "float16"):
        """
        Intialize the indexer
        """
        if n_subquantizers > 0:
            self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(vector_sz)
        self.index_id_to_db_id = []
        self.dtype = dtype

    def to_gpu(self):
        logger.info(f"Moving index to gpu, found {faiss.get_num_gpus()} GPUs")
        resources = [faiss.StandardGpuResources() for i in range(faiss.get_num_gpus())]
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True if self.dtype == "float16" else False
        co.shard = True
        index_gpu = faiss.index_cpu_to_gpu_multiple_py(resources, self.index, co=co)
        self.index = index_gpu

    def index_data(self, ids: list, embeddings: np.array):
        self._update_id_mapping(ids)
        embeddings = embeddings.astype(self.dtype)
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)

        logger.info(f'Total data indexed {len(self.index_id_to_db_id)}')

    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size: int = 2048) -> List[Tuple[List[object], np.array]]:
        """
        Returns a list of tuples containing the external ids and scores of the top documents (np array of float32)
        """
        query_vectors = query_vectors.astype(self.dtype)
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        for k in tqdm(range(nbatch), desc="searching knn"):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        logger.info("Finished searching knn")
        return result

    def serialize(self, dir_path: str):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        logger.info(f'Serializing index to {index_file}, meta data to {meta_file}')

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path: str):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        logger.info(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)
        logger.info('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        self.index_id_to_db_id.extend(db_ids)
