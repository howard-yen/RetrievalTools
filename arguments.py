from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict
from pathlib import Path
import glob

# TODO: break into encoder and generator options?
@dataclass
class ModelOptions:
    model_name_or_path: str = None
    seed: int = 42

    # Generation settings
    temperature: float = 0.0 
    top_p: float = 1.0
    do_sample: bool = False
    input_max_length: int = 8192 # also the passage max length for when encoding
    generation_max_length: int = 1024
    generation_min_length: int = 0
    stop_newline: bool = False

    # local models
    no_torch_compile: bool = False
    no_bf16: bool = False
    rope_theta: Optional[float] = None
    use_chat_template: bool = False
    use_vllm: bool = False
    batch_size: int = 512 # per gpu batch size
    use_hf: bool = True

    # encoder settings
    pooling: str = "mean" # other options: "cls", "max", "last", "mean"
    query_prompt: str = "" # text to preprend to the query
    passage_prompt: str = "" # text to prepend to the passage
    normalize_embedding: bool = False

    def __post_init__(self):
        assert self.model_name_or_path is not None, "model_name_or_path must be set"


"""
For data.Corpus
"""
@dataclass
class CorpusOptions:
    # You may start with an empty corpus if you only want to load specific ids later
    paths: List[Union[Path, str]] = field(default_factory=list)
    # maximum number of whitespace words
    chunk_size: int = 256
    # minimum number of whitespace words
    threshold_chunk_size: int = 32
    # key name for the id field
    id_field: str = "id"
    # key name for the text field
    text_field: str = "text"
    # additional metadata fields to get from the original corpus
    metadata_fields: List[str] = field(default_factory=list)

    num_workers: int = 8
    count_only: bool = False
    corpus_type: str = "chunk" # "chunk" or "normal"

    def __post_init__(self):
        self.paths = [item for sublist in self.paths for item in glob.glob(sublist)]


@dataclass
class ShardOptions:
    num_shards: int = 1
    shard_id: int = 0

    def __post_init__(self):
        if self.num_shards <= 0:
            raise ValueError("num_shards must be greater than 0")
        if self.shard_id < 0 or self.shard_id >= self.num_shards:
            raise ValueError("shard_id must be between 0 and num_shards")


# FAISS Index
@dataclass
class IndexOptions:
    n_subquantizers: int = 0 # Number of subquantizer used for vector quantization, if 0 flat index is used
    n_bits: int = 8 # Number of bits per subquantizer
    batch_size: int = 2048 # Batch size for index search
    no_fp16: bool = False # Inference in fp32 instead of fp16 (not necessary most times)
    embedding_files: List[str] = field(default_factory=list)


@dataclass
class RetrievalDataSingle:
    input_path: str = ""
    # if set to none, use output_dir/model_name/basename of input path + .jsonl
    output_path: str = None 
    query_field: str = "query"


@dataclass
class RetrievalDataOptions:
    datasets: List[Dict[str, str]] = field(default_factory=list)
    topk: int = 100

    def __post_init__(self):
        new_datasets = []
        for dataset in self.datasets:
            if "input_path" not in dataset:
                raise ValueError("Input path missing in dataset")
            if "query_field" not in dataset:
                raise ValueError("Query field missing in dataset")
            new_datasets.append(RetrievalDataSingle(**dataset))
        self.datasets = new_datasets


@dataclass
class RetrieverOptions:
    retriever_type: str = "dense"
    include_corpus: bool = False
    port: int = 8000