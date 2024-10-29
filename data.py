import os
import json

from typing import Optional, List, Union

from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from datatools import load
from multiprocessing import Pool

import logging
logger = logging.getLogger(__name__)

class Corpus():
    """
    Specifically designed for retrieval purposes, with datatools and huggingface's dataset as the backbones

    one important operation to support is fast lookup -- given the passage id, we need to be able to get the text
    the structure of the id looks like this: "{path}/{index}/{paragraph_index}"
    the paragraph_index needs to be deterministic, and we must use the same algorithm to generate when building the index and when querying
 
    TODO: 
     - do we really need to load everything into memory? an alternative approach is to yield the chunks as needed
     - get statistics about the chunks
     - add support for just a single file lol
     - support for no chunking
    """
    
    def __init__(self, *paths: List[Union[Path, str]], num_workers: int = 8, count_only: bool = False):
        # this defines the number of words in each chunk, should not change it after the index is built
        self.CHUNK_SIZE = 512
        self.field = "text"

        self.paths = set()
        self.chunks = {}
        # with Pool(num_workers) as pool:
        #     mappings = pool.map(self.add_path, paths)
        self.total_lines = 0
        self.total_chunks = 0

        self.count_only = count_only

        if num_workers <= 1:
            mappings = [self.add_path(p) for p in paths]
        else:
            mappings = process_map(self.add_path, paths, max_workers=num_workers)

        for num_lines, num_chunks, m in mappings:
            self.total_lines += num_lines
            self.total_chunks += num_chunks
            self.chunks.update(m)
    

    def add_path(self, path: Union[Path, str]):
        self.paths.add(path)
        data = load(path)
        mappings = {}
        num_chunks = 0
        for i, d in enumerate(tqdm(data, leave=False, desc=f"Processing {path}")):
            chunks = self.process_text(d[self.field])
            num_chunks += len(chunks)

            if not self.count_only:
                mapping = {f"{path}/{i}/{j}": k for j, k in enumerate(chunks)}
                mappings.update(mapping)

        return len(data), num_chunks, mappings
                

    def process_text(self, text: str):
        paragraphs = text.split("\n")
        paragraphs = [x for x in paragraphs if x]

        # now break each paragraph into chunks of at most CHUNK_SIZE words
        chunks = []
        for p in paragraphs:
            words = p.split(" ")
            for i in range(0, len(words), self.CHUNK_SIZE):
                lens = len(words[i:i+self.CHUNK_SIZE])
                chunks.append((" ".join(words[i:i+self.CHUNK_SIZE]), lens))
        
        # now combine chunks that are fewer than CHUNK_SIZE words if possible
        combined_chunks = []
        cur_chunk = []
        cur_len = 0
        for c in chunks:
            if cur_len + c[1] > self.CHUNK_SIZE:
                combined_chunks.append(" ".join(cur_chunk))
                cur_chunk = [c[0]]
                cur_len = c[1]
            else:
                cur_chunk.append(c[0])
                cur_len += c[1]

        return combined_chunks

    
    def get_text(self, id: str) -> str:
        fields = id.split("/")
        path = "/".join(fields[:-2])
        if path not in self.paths:
            logger.info(f"Path {path} not found in the corpus, adding.")
            length, num_chunks, m = self.add_path(path)
            self.total_lines += length
            self.total_chunks += num_chunks
            self.chunks.update(m)

        # we expect the id to always be in the corpus
        return self.chunks[id]

    
    def get_text_greedy(self, id: str) -> str:
        fields = id.split("/")
        path = "/".join(fields[:-2])
        line_num = int(fields[-2])

        if id not in self.chunks:
            data = load(path)
            chunks = self.process_text(data[line_num][self.field])
            self.chunks.update({f"{path}/{line_num}/{i}": c for i, c in enumerate(chunks)})
        
        return self.chunks[id]


    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def size(self) -> int:
        return len(self.chunks)
    
    def get_chunks(self):
        return self.chunks
