import pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List

from tqdm import trange, tqdm
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding
from sentence_transformers import SentenceTransformer

from data import CorpusDataset, LengthSortedCollator
from utils import mean_pooling, max_pooling, cls_pooling, last_token_pooling

import logging
logger = logging.getLogger(__name__)

class Retriever():
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
        self.torch_dtype = dtype if dtype in ["auto", None] else getattr(torch, dtype)
    
    def encode(self, context: List[str], prompt: str = "") -> np.array:
        raise NotImplementedError

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError

class STRetriever(Retriever):
    # sentence transformer retriever, preferred due to multi-gpu support
    # recommend using torch dtype = float16
    def __init__(self, 
        model_name_or_path: str, 
        ctx_size: Optional[int] = None, 
        batch_size: int = 32, 
        dtype: str = "float16",
        normalize_embedding: bool = True,
        **model_kwargs,
    ):
        super().__init__(model_name_or_path, ctx_size, batch_size, dtype)
        self.model = SentenceTransformer(
            model_name_or_path, 
            device="cuda", 
            trust_remote_code=True,
            model_kwargs={"trust_remote_code": True, "torch_dtype": self.torch_dtype, **model_kwargs},
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


class HFRetriever(Retriever):
    def __init__(
        self, 
        model_name_or_path: str, 
        ctx_size: Optional[int] = None, 
        batch_size: int = 32, 
        dtype: str = "float16",
        pooling: str = "mean", # other options: "cls", "max", "last"
        normalize_embedding: bool = True,
        **model_kwargs
    ):
        super().__init__(model_name_or_path, ctx_size, batch_size, dtype)
        assert pooling in ["mean", "cls", "max", "last"], "Pooling must be one of 'mean', 'cls', 'max', 'last'"
        self.pooling = pooling
    
        self.config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, 
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            **model_kwargs,
        ).eval().to("cuda")
        
        if ctx_size is None:
            self.ctx_size = self.config.max_position_embeddings
        elif ctx_size > self.config.max_position_embeddings:
            logger.warning(f"ctx_size {ctx_size} is greater than model max_seq_length {self.model.max_position_embeddings}, this may return unexpected results")

        self.normalize_embedding = normalize_embedding

    @torch.no_grad()
    def encode(self, context: List[str], prompt: str = "") -> np.array:
        """
        Returns an np.array
        """
        allemb = []
        dataset = CorpusDataset(context, self.tokenizer, self.ctx_size, prompt)
        # This collator first sort the input by the token length so that we don't have to pad as much
        collator = LengthSortedCollator(self.tokenizer, mini_batch_size=self.batch_size)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size*128,
            shuffle=False, 
            num_workers=8, 
            pin_memory=True, 
            collate_fn=collator
        )

        # for i in trange(0, len(context), self.batch_size):
            # batch = context[i:i+self.batch_size]
            # batch = [prompt + text for text in batch]
            # inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.ctx_size)

        # print("Iterating over dataloader")
        for batch_idx, (batch, length_idxs) in enumerate(tqdm(dataloader)):
            
            batch_embeddings = []
            for mini_batch in tqdm(batch, leave=False):
                inputs = mini_batch.to(self.model.device)
                # print(inputs.input_ids.shape, inputs.input_ids.numel() - inputs.attention_mask.sum())
                outputs = self.model(**inputs.to(self.model.device))

                if self.pooling == "mean":
                    embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                elif self.pooling == "cls":
                    embeddings = cls_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                elif self.pooling == "max":
                    embeddings = max_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                elif self.pooling == "last":
                    embeddings = last_token_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                else:
                    raise ValueError(f"Pooling {self.pooling} not supported")

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


def load_retriever(
    model_name_or_path: str, 
    max_length: int = 512,
    batch_size: int = 32,
    use_hf: bool = False,
):
    if use_hf:
        kwargs = {}
        if "gte" in model_name_or_path:
            kwargs = {}
            if 'gte-qwen2-1.5b-instruct' in model_name_or_path.lower():
                kwargs['pooling'] = 'last'
                kwargs["attn_implementation"] = "flash_attention_2"
            elif 'gte-large-en-v1.5' in model_name_or_path.lower():
                kwargs['pooling'] = 'cls'
                kwargs.update({"unpad_inputs": True, "use_memory_efficient_attention": True})
        return HFRetriever(model_name_or_path, ctx_size=max_length, batch_size=batch_size, **kwargs)
    
    logger.warning("Using sentence transformer; we highly recommend using HF retriever for better performance")
    kwargs = {"attn_implementation": "flash_attention_2"}
    return STRetriever(model_name_or_path, ctx_size=max_length, batch_size=batch_size)
