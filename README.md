# <img src="assets/rt.png" alt="RT" width="30"> RetrievalTools


This repository contains a collection of useful retrieval tools, inspired by previous codebases, such as [Contriever](https://github.com/facebookresearch/contriever).
The main difference from previous repos like Contriever and DPR-Scale is the additional support for different types of retrievers like BM25, SentenceTransformers, and API calls. 
<!-- We also minimize the amount of extra scaffolding, such as Hydra configurations, to make the codebase more accessible to newcomers. -->
The goal of this repository is to provide a simple, easy-to-use, and efficient codebase for retrieval tasks.

We support the following operations:
 - Building large-sclae dense indices with sharding
 - Retrieving across sharded dense indices on GPUs and merging the results
 - Common API for different retrieval models (e.g., BM25, SentenceTransformers, API calls)
 <!-- - Retrieval-from-context with different chunking stratgies -->


The code is especially customized towards a slurm-based cluster, and we optimize for easy paralleization and efficient memory usage.
This can be useful for large-scale retrieval tasks, where we might want to annotate a pre-training scale corpus with retrieval results.
We also provide a simple API for playing around with different retrieval models by enabling serving indices.

> [!NOTE]
> This repository is still under development, please be patient as I'm working on adding more features and improving the documentation! Check the TODOs section for more information.

## Overview

### Key Classes

- `Encoder`: Fast encoding with dense models
- `Corpus`: A collection of documents with ids, text, and other metadata
- `Index`: A dense index that supports vector search with FAISS
- `Retriever`: A retrieval model that supports different retrieval models


## Installation

We highly recommend using [pixi](https://pixi.prefix.dev/latest/) and [uv](https://docs.astral.sh/uv/) to manage the dependencies.
If you are looking to serve dense indices with FAISS, then you can set up the environment with:

```bash
pixi install
pixi shell # to activate the environment
```

You may want to double check that you can import everything correctly:
```python
import faiss
faiss.GpuClonerOptions # check that we can import this
print(faiss.get_num_gpus()) # check that this is not 0 when there is a GPU available
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from transformers import AutoModel, AutoTokenizer, AutoConfig
```

If you don't need the `faiss-gpu` package, we recommend using uv:
```bash
uv sync
# or if you prefer to use just pip, do the following in your environment
pip install -e . 
```


Then, you should be able to import the package with:
```python
import retrievaltools as rt

# load the retriever
retriever = rt.load_retriever(rt.RetrieverOptions(retriever_type="web_search", cache_path="cache/serper.json"))
```


## Encoding

The stages of retrieval are as follows for single-vector dense retrievers (e.g., DPR, Contriever, etc.):
1. **Embedding the corpus** (`generate_passage_embeddings.py`): Encode the corpus into dense vectors
2. **Retrieve for queries** (`passage_retrieval.py`): For each query, retrieve the top-k passages from the corpus
3. **Add text** (`text_annotation.py`): Optionally, add the passage text to the retrieved results if only the passage ids were stored.


### Small Corpora

For smaller corpora (e.g., Wikipedia which has ~20M passages), you may follow these steps, using Wikipedia as an example:

1. **Prepare the corpus**: Put the corpus that you want to encode in file, supported file formats are `.tsv` and `.jsonl`.
In this example, we use the Wikipedia dump, which can be downloaded from the DPR repository.
```bash
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gunzip psgs_w100.tsv.gz
```

2. **Encode the corpus**: Use the `generate_passage_embeddings.py` script to encode the corpus. You may see `scripts/gen_emb.sh` for an example.
```bash
# remember to activate your environment
python generate_passage_embeddings.py \
    --config configs/models/qwen3-0.6b.yaml configs/corpus/wiki.yaml \
    --output_dir embeddings/qwen3-0.6b/wikipedia \
    --output_prefix wiki \
    --shard_id 0 --num_shards 100 \
    --input_max_length 512 \
    --batch_size 256 
```

In the config files, you may specify things like:
```yaml
# wiki corpus configs
corpus_options:
  paths: # support global/local paths and glob patterns
    - data/psgs_w100.tsv
    - data/wiki*.jsonl.gz
  text_field: text
  metadata_fields: [title]
  num_workers: 8

# qwen model config

```

### Large Corpora 

When building a large retrieval corpus (>500M documents or >100B tokens), it is often necessary to shard the corpus and parallelize the encoding process.

## Indexing

We support indexing with FAISS, and with paralleization across GPUs.



## TODOs

- [] Save passage text instead of loading from file

## Contact

Please email me at `hyen@cs.princeton.edu` if you run into any issues or have any questions.
