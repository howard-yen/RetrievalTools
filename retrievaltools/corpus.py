import os
import json

from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import PreTrainedTokenizerBase

import datasets
from torch.utils.data import Dataset, DataLoader
import datatools
from datatools.io_utils import zstd_utf8_read_open, Subset

from retrievaltools.arguments import CorpusOptions, ShardOptions
from retrievaltools.utils import get_shard_idx, init_logger

logger = init_logger(__name__)


class Corpus():
    """
    Assuem data is already processed: each instance in the dataset corresponds to a document.
    """

    def __init__(
        self, 
        paths: List[Union[Path, str]] = [], 
        id_field: str = "id",
        text_field: str = "text", 
        metadata_fields: List[str] = [], 
        num_workers: int = 8, 
        shard_options: Optional[ShardOptions] = None
    ):
        self.paths = paths
        self.text_field = text_field
        self.metadata_fields = metadata_fields
        self.id_field = id_field
        self.num_workers = num_workers

        self.data: Dict[str, Dict[str, Any]] = {}
        if shard_options is not None:
            self.shard_options = shard_options
        else:
            self.shard_options = ShardOptions(num_shards=1, shard_id=0)

        # load the paths and shard accordingly
        if self.shard_options.shard_files:
            assert len(paths) >= self.shard_options.num_shards, "If shard_files is True, then the number of paths must be >= the number of shards"
            start_idx, end_idx = get_shard_idx(len(paths), self.shard_options.num_shards, self.shard_options.shard_id)
            self.paths = self.paths[start_idx:end_idx]

        self.num_workers = num_workers
        data = datatools.load(*self.paths)
        if not self.shard_options.shard_files:
            data = Subset.shard(data, num_shards=self.shard_options.num_shards, shard_id=self.shard_options.shard_id)
            
        self.data = {}
        for i, d in enumerate(data):
            if self.id_field is None:
                sample_id = i
            else:
                sample_id = d[self.id_field]
            self.data[sample_id] = {"text": d[self.text_field], **{k: d[k] for k in self.metadata_fields}}
        
        logger.info(f"Loaded {len(self.data)} documents in the corpus")
        

    def get_item(self, id: str):
        return self.data[id]
    
    
    def get_text(self, id: str) -> str:
        return self.data[id][self.text_field]


    def __len__(self):
        return len(self.data)


    @property
    def size(self):
        return len(self.data)


    def get_data(self):
        return self.data

 
class ChunkedCorpus():
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
    
    def __init__(
        self,
        paths: List[Union[Path, str]] = [],
        chunk_size: int = 256, 
        threshold_chunk_size: int = 32,
        text_field: str = "text",
        metadata_fields: List[str] = [],
        num_workers: int = 8, 
        shard_options: Optional[ShardOptions] = None,
        count_only: bool = False
    ):
        # this defines the number of words in each chunk, should not change it after the index is built
        self.chunk_size: int = chunk_size

        # if a chunk is less than this, and it's left at the end, we exclude it 
        self.threshold_chunk_size: int = threshold_chunk_size
        self.field: str = text_field
        self.metadata_fields: List[str] = metadata_fields

        # if text is in the metadata fields, we need to rename it to "full_text", since there may be instances
        # where we want both the chunked text ("text") and the full text ("full_text")
        # now if there is a full_text field, we should warn the user... 
        if "full_text" in self.metadata_fields:
            logger.warning("full_text is in the metadata fields, this will actually take from the text field")
        if "text" in self.metadata_fields:
            self.metadata_fields = [f for f in self.metadata_fields if f != "text"] + ["full_text"]

        self.paths: Set[Union[Path, str]] = set()
        # chunks maps from id (path/line_idx/chunk_idx) to Dict[str, Any] (text, metadata)
        self.chunks: Dict[str, Dict[str, Any]] = {}
        self.total_lines: int = 0
        self.total_chunks: int = 0

        if shard_options is not None:
            self.shard_options = shard_options
        else:
            self.shard_options = ShardOptions(num_shards=1, shard_id=0)

        self.count_only: bool = count_only

        self.num_workers = num_workers

        if num_workers <= 1:
            mappings = [self.add_path(p) for p in paths]
        else:
            mappings = process_map(self.add_path, paths, max_workers=num_workers, desc="Adding paths to corpus")

        for num_lines, num_chunks, m in mappings:
            self.total_lines += num_lines
            self.total_chunks += num_chunks
            self.chunks.update(m)
    

    def add_metadata(self, data: Dict[str, Any], chunks: List[str]) -> List[Dict[str, Any]]:
        if "full_text" in self.metadata_fields:
            data["full_text"] = data.pop("text")

        return [{"text": c, **{k: data[k] for k in self.metadata_fields}} for c in chunks]


    def add_path(self, path: Union[Path, str]) -> Tuple[int, int, Dict[str, str]]:
        self.paths.add(path)
        data = datatools.load(path)
        # mapping maps from id (path/line_idx/chunk_idx) to Dict[str, Any] (text, metadata)
        mappings: Dict[str, Dict[str, Any]] = {}
        num_chunks: int = 0
        start_idx, end_idx = get_shard_idx(len(data), self.shard_options.num_shards, self.shard_options.shard_id)

        if isinstance(data, datasets.Dataset):
            data = data.select(range(start_idx, end_idx))
        else:
            data = data[start_idx:end_idx]

        for line_idx, d in enumerate(tqdm(
            data, leave=False, desc=f"Processing {path}", total=len(data), disable=self.num_workers > 1
        )):
            line_idx += start_idx # we need to offset the line_idx by the start_idx so later access is correct
            if d[self.field] is None:
                continue
            chunks = self.process_text(d[self.field])
            chunks = self.add_metadata(d, chunks)
            num_chunks += len(chunks)

            if not self.count_only:
                mapping = {f"{path}/{line_idx}/{chunk_idx}": chunk for chunk_idx, chunk in enumerate(chunks)}
                mappings.update(mapping)

        return len(data), num_chunks, mappings
                

    def process_text(self, text: str):
        """
        It's really important for this function to be deterministic, so that we can construct identical texts when building the index and retrieving from it later.
        Currently, we are only using whitespace tokenizing. Maybe I will switch to spacy or something later on, it would be too slow to use an actual tokenizer because it would be super slow (but it could be worth benchmarking tiktoken).
        """
        paragraphs = text.split("\n")
        paragraphs = [x for x in paragraphs if x]

        # now break each paragraph into chunks of at most CHUNK_SIZE words
        chunks = []
        for p in paragraphs:
            # skip empty paragraphs
            p = p.strip()
            if not p:
                continue

            words = p.split()
            for i in range(0, len(words), self.chunk_size):
                length = len(words[i:i+self.chunk_size])
                chunks.append((" ".join(words[i:i+self.chunk_size]), length))
            chunks[-1] = (chunks[-1][0] + "\n", chunks[-1][1])
        
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
        
        # add the last chunk if it is not too short, otherwise discard it
        if cur_len >= self.threshold_chunk_size:
            combined_chunks.append(" ".join(cur_chunk))

        return combined_chunks

    
    def get_item(self, id: str) -> Dict[str, Any]:
        """
        Process the entire file/path and add it to our corpus
        """
        fields = id.split("/")
        path = "/".join(fields[:-2])
        if path not in self.paths:
            logger.info(f"Path {path} not found in the corpus, adding.")
            length, num_chunks, new_chunks = self.add_path(path)
            self.total_lines += length
            self.total_chunks += num_chunks
            self.chunks.update(new_chunks)

        # we expect the id to always be in the corpus
        return self.chunks[id]


    def get_item_greedy(self, id: str) -> Dict[str, Any]:
        """
        Greedy only process the line in the file that we want to process
        """
        fields = id.split("/")
        path = "/".join(fields[:-2])
        line_num = int(fields[-2])

        if id not in self.chunks:
            # avoid loading the entire file into memory
            data = {}
            if ".jsonl" in path:
                path = Path(path)
                if path.suffixes[-1] in [".zstd", ".zst"]:
                    with zstd_utf8_read_open(path) as f:
                        for i, line in enumerate(f):
                            if i == line_num:
                                data = json.loads(line)
                                break
                else:
                    with open(path, "r") as f:
                        for i, line in enumerate(f):
                            if i == line_num:
                                data = json.loads(line)
                                break
                chunks = self.process_text(data[self.field])

            else:
                data = datatools.load(path)
                chunks = self.process_text(data[line_num][self.field])

            chunks = self.add_metadata(data, chunks)
            self.chunks.update({f"{path}/{line_num}/{i}": c for i, c in enumerate(chunks)})
        
        return self.chunks[id]


    def __len__(self) -> int:
        return len(self.chunks)


    @property
    def size(self) -> int:
        return len(self.chunks)
    

    def get_data(self):
        return self.chunks


def load_corpus(corpus_options: CorpusOptions, shard_options: Optional[ShardOptions] = None):
    if corpus_options.corpus_type == "chunk":
        return ChunkedCorpus(
            paths=corpus_options.paths,
            chunk_size=corpus_options.chunk_size,
            threshold_chunk_size=corpus_options.threshold_chunk_size,
            text_field=corpus_options.text_field,
            metadata_fields=corpus_options.metadata_fields,
            shard_options=shard_options,
        )
    else:
        return Corpus(
            paths=corpus_options.paths,
            id_field=corpus_options.id_field,
            text_field=corpus_options.text_field,
            metadata_fields=corpus_options.metadata_fields,
            shard_options=shard_options,
        )


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
        # we have a list of dict from the dataset, we first sort them by length (keeping the original indices) and then collate them into smaller batches
        lengths = [-len(x["input_ids"]) for x in batch]
        length_sorted_idx = np.argsort(lengths)
        batch_sorted = [batch[i] for i in length_sorted_idx]

        mini_batches = [
            self.tokenizer.pad(batch_sorted[i:i+self.mini_batch_size], padding="longest", return_tensors="pt") 
            for i in range(0, len(batch_sorted), self.mini_batch_size)
        ]
        return mini_batches, length_sorted_idx
