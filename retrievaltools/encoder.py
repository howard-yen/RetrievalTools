import pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List

from tqdm import trange, tqdm
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding
from vllm import LLM

from retrievaltools.corpus import CorpusDataset, LengthSortedCollator
from retrievaltools.utils import POOLING_FUNC, init_logger
from retrievaltools.arguments import ModelOptions

logger = init_logger(__name__)

class Encoder():
    """
    Encoder class for encoding text into embeddings
    The main functionality to to implement encode()
    encode takes a list of strings and returns an np.array of embeddings

    Args:
        model_name_or_path: str
            The name of the model to use
        ctx_size: Optional[int]
            The maximum number of tokens to encode
        batch_size: int
            The batch size to use for encoding
        dtype: str
            The dtype to use for encoding

    TODOs:
    - Add support for API encoding
    """
    def __init__(
        self,
        model_name_or_path: str,
        ctx_size: Optional[int] = None,
        batch_size: int = 32,
        dtype: str = "float16",
    ):
        self.model_name_or_path = model_name_or_path
        self.ctx_size = ctx_size
        self.batch_size = batch_size
        self.dtype = dtype
        self.dtype = dtype if dtype in ["auto", None] else getattr(torch, dtype)
    
    def encode(self, context: List[str], prompt: str = "") -> np.array:
        raise NotImplementedError

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError


class STEncoder(Encoder):
    """
    Sentence Transformer encoder
    recommend using torch dtype = float16
    """
    def __init__(self, 
        model_name_or_path: str, 
        ctx_size: Optional[int] = None, 
        batch_size: int = 32, 
        dtype: str = "float16",
        normalize_embedding: bool = True,
        **model_kwargs,
    ):
        from sentence_transformers import SentenceTransformer
        super().__init__(model_name_or_path, ctx_size, batch_size, dtype)
        self.model = SentenceTransformer(
            model_name_or_path, 
            device="cuda", 
            trust_remote_code=True,
            model_kwargs={"trust_remote_code": True, "dtype": self.dtype, **model_kwargs},
            tokenizer_kwargs={"trust_remote_code": True}
        ).eval()
        self.tokenizer = self.model.tokenizer
        
        if ctx_size is not None:
            self.model.max_seq_length = ctx_size
            if ctx_size > self.model.max_seq_length:
                logger.warning(f"ctx_size {ctx_size} is greater than model max_seq_length {self.model.max_seq_length}, this may return unexpected results")

        self.normalize_embedding = normalize_embedding

    @torch.no_grad()
    def encode(self, context: List[str], prompt: str = "") -> np.array:
        """
        Prompt is prepended to the context
        Returns an np.array
        """
        if len(context) > 1000 and torch.cuda.device_count() > 1:
            pool = self.model.start_multi_process_pool()
            emb = self.model.encode_multi_process(
                context, 
                pool, 
                normalize_embeddings=self.normalize_embedding, 
                prompt=prompt, 
                batch_size=self.batch_size,
            )
            self.model.stop_multi_process_pool(pool)
        else:
            emb = self.model.encode(
                context, 
                normalize_embeddings=self.normalize_embedding, 
                prompt=prompt, 
                show_progress_bar=True, 
                convert_to_numpy=True, 
                batch_size=self.batch_size,
            )
        return emb
    
    @property
    def hidden_size(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class HFEncoder(Encoder):
    """
    Hugging Face encoder, we have some optimization that makes encoding faster (by efficient batching)
    recommend using torch dtype = float16
    """
    def __init__(
        self, 
        model_name_or_path: str, 
        ctx_size: Optional[int] = None, 
        batch_size: int = 32, 
        dtype: str = "float16",
        pooling: str = "mean", # other options: "cls", "max", "last"
        normalize_embedding: bool = False,
        tqdm: bool = True,
        **model_kwargs
    ):
        super().__init__(model_name_or_path, ctx_size, batch_size, dtype)
        assert pooling in POOLING_FUNC, "Pooling must be one of 'mean', 'cls', 'max', 'last'"
        self.pooling = POOLING_FUNC[pooling]

        self.config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, 
            dtype=self.dtype,
            trust_remote_code=True,
            device_map="auto",
            **model_kwargs,
        ).eval()

        # self.model.to("cuda")
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model, output_device=torch.device("cpu")) # can't put cpu but if all outputs go to one gpu, it may run out of memory
        #     self.batch_size *= torch.cuda.device_count()
        
        if ctx_size is None:
            logger.info(f"Setting ctx_size to model max_position_embeddings {self.config.max_position_embeddings}")
            self.ctx_size = self.config.max_position_embeddings
        elif ctx_size > self.config.max_position_embeddings:
            logger.warning(f"ctx_size {ctx_size} is greater than model max_seq_length {self.model.max_position_embeddings}, this may return unexpected results")

        self.normalize_embedding = normalize_embedding
        self.tqdm = tqdm


    @torch.no_grad()
    def encode(self, context: List[str], prompt: str = "") -> np.array:
        """
        Returns an np.array
        """
        allemb = []
        dataset = CorpusDataset(context, self.tokenizer, self.ctx_size, prompt)
        # This collator first sort the input by the token length so that we don't have to pad as much
        collator = LengthSortedCollator(self.tokenizer, mini_batch_size=self.batch_size)
        # we sort the mini-batch within a larger batch
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size*128,
            shuffle=False, 
            num_workers=8, 
            pin_memory=True, 
            collate_fn=collator
        )

        for batch_idx, (batch, length_idxs) in enumerate(tqdm(dataloader, desc="Encoding", disable=not self.tqdm)):
            batch_embeddings = []
            for mini_batch in tqdm(batch, leave=False, desc="Mini-batch", disable=not self.tqdm):
                device = self.model.module.device if hasattr(self.model, "module") else self.model.device
                # device = "cpu"
                inputs = mini_batch.to(device)
                outputs = self.model(**inputs)

                embeddings = self.pooling(outputs.last_hidden_state, inputs["attention_mask"])

                if self.normalize_embedding:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                del inputs
                batch_embeddings.append(embeddings.cpu().numpy())

            batch_embeddings = np.concatenate(batch_embeddings, axis=0)
            original_embeddings = np.zeros_like(batch_embeddings)
            original_embeddings[length_idxs] = batch_embeddings

            allemb.append(original_embeddings)

        return np.concatenate(allemb, axis=0)
    
    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size


class VLLMEncoder(Encoder):
    def __init__(
        self,
        model_name_or_path: str,
        ctx_size: Optional[int] = None,
        batch_size: int = 32,
        dtype: str = "float16",
    ):
        super().__init__(model_name_or_path, ctx_size, batch_size, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = LLM(
            model=model_name_or_path, 
            task="embed",
            enforce_eager=True,
            dtype=self.torch_dtype,
        )
    
    def encode(self, context: List[str], prompt: str = "") -> np.array:
        allemb = []

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError


def load_encoder(
    model_options: ModelOptions,
) -> Encoder:

    if model_options.use_hf:
        kwargs = {}
        if "gte" in model_options.model_name_or_path:
            kwargs = {}
            # kwargs["attn_implementation"] = "flash_attention_2"
            if 'gte-large-en-v1.5' in model_options.model_name_or_path.lower():
                kwargs.update({"unpad_inputs": True, "use_memory_efficient_attention": True})

        return HFEncoder(
            model_options.model_name_or_path, 
            ctx_size=model_options.input_max_length, 
            batch_size=model_options.batch_size, 
            pooling=model_options.pooling,
            normalize_embedding=model_options.normalize_embedding,
            tqdm=model_options.tqdm,
            **kwargs
        )
    
    logger.warning("Using sentence transformer; we highly recommend using HF retriever for better performance")
    kwargs = {"attn_implementation": "flash_attention_2"}
    return STEncoder(
        model_options.model_name_or_path, 
        ctx_size=model_options.input_max_length, 
        batch_size=model_options.batch_size, 
        pooling=model_options.pooling,
        **kwargs
    )
