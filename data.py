import os
import json

from pathlib import Path
from typing import Optional, List, Union
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import PreTrainedTokenizerBase

from torch.utils.data import Dataset, DataLoader

from datatools import load

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
    
    def __init__(self, *paths: List[Union[Path, str]], chunk_size=256, num_workers: int = 8, count_only: bool = False):
        # this defines the number of words in each chunk, should not change it after the index is built
        self.chunk_size = chunk_size

        # if a chunk is less than this, and it's left at the end, we exclude it 
        self.threshold_chunk_size = 32 

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
        paragraphs = [x+"\n" for x in paragraphs if x]

        # now break each paragraph into chunks of at most CHUNK_SIZE words
        chunks = []
        for p in paragraphs:
            words = p.split(" ")
            for i in range(0, len(words), self.chunk_size):
                length = len(words[i:i+self.chunk_size])
                chunks.append((" ".join(words[i:i+self.chunk_size]), length))
        
        # now combine chunks that are fewer than CHUNK_SIZE words if possible
        combined_chunks = []
        cur_chunk = []
        cur_len = 0
        for c in chunks:
            if cur_len + c[1] > self.chunk_size:
                if cur_len >= self.threshold_chunk_size:
                    # discard the current chunk if it is too short 
                    # (this happens in cases where there is a long paragraph with a couple words left over, and then next paragraph is too long)
                    combined_chunks.append(" ".join(cur_chunk))

                cur_chunk = [c[0]]
                cur_len = c[1]
            else:
                cur_chunk.append(c[0])
                cur_len += c[1]
        
        # add the last chunk if it is not too short
        if cur_len >= self.threshold_chunk_size:
            combined_chunks.append(" ".join(cur_chunk))

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


class CorpusDataset(Dataset):
    def __init__(self, texts, tokenizer, ctx_size, prompt=""):
        self.texts = texts
        self.tokenizer = tokenizer
        self.ctx_size = ctx_size
        self.prompt = prompt
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.tokenizer(self.prompt + self.texts[idx], truncation=True, max_length=self.ctx_size)

@dataclass
class LengthSortedCollator:
    tokenizer: PreTrainedTokenizerBase
    mini_batch_size: int = 32
    
    def __call__(self, batch):
        # we have a list of dict from above, we first sort them by length (keeping the original indices) and then collate them into smaller batches
        lengths = [-len(x["input_ids"]) for x in batch]
        length_sorted_idx = np.argsort(lengths)
        batch_sorted = [batch[i] for i in length_sorted_idx]

        mini_batches = [
            self.tokenizer.pad(batch_sorted[i:i+self.mini_batch_size], padding="longest", return_tensors="pt") 
            for i in range(0, len(batch_sorted), self.mini_batch_size)
        ]
        return mini_batches, length_sorted_idx
