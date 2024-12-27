from dataclasses import dataclass, field
from typing import Optional, List, Union
from pathlib import Path

@dataclass
class ModelOptions:
    model_name_or_path: str = None
    seed: int = 42

    # Generation settings
    temperature: float = 0.0 
    top_p: float = 1.0
    do_sample: bool = False
    input_max_length: int = 8192
    generation_max_length: int = 1024
    generation_min_length: int = 0
    stop_newline: bool = False

    # local models
    no_torch_compile: bool = False
    no_bf16: bool = False
    rope_theta: Optional[float] = None
    use_chat_template: bool = False
    use_vllm: bool = False

    # def __post_init__(self):
    #     if self.model_name_or_path is None:
    #         raise ValueError("model_name_or_path must be provided")

@dataclass
class CorpusOptions:
    paths: Optional[List[Union[Path, str]]] = field(default_factory=list)
    chunk_size: int = 256
    threshold_chunk_size: int = 32
    num_workers: int = 8
    count_only: bool = False

@dataclass
class ShardOptions:
    num_shards: int = 1
    shard_id: int = 0

    def __post_init__(self):
        if self.num_shards <= 0:
            raise ValueError("num_shards must be greater than 0")
        if self.shard_id < 0 or self.shard_id >= self.num_shards:
            raise ValueError("shard_id must be between 0 and num_shards")