# <img src="https://github.com/user-attachments/assets/55a90c30-4790-49c7-92f6-b186ec915724" alt="RT" width="30"> RetrievalTools


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

We use FAISS to support different types of dense indices; however, it can be tricky to get the environment exactly right.
In practice, I find it easier to use a separate conda environment specifically for running FAISS GPU.

### Without FAISS

For simple encoding, you may not need FAISS, and you can install all packages with:

```bash
pip install -r requirements.txt
```

You should install `torch` following [these instructions](https://pytorch.org/get-started/locally/) to match your CUDA version.

### FAISS

FAISS is critical for the retrieval step if you are using a dense index; it is responsible for fast indexing and supports many useful functions (e.g., quantization, multi-gpu index, etc.).
To install the package, you should set up a conda environment and install PyTorch and FAISS (guide [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)).
Additionally, you should install `transformers` and `sentence-transformers`.
You also want to install `datatools` [here](https://github.com/CodeCreator/datatools).

## Usage Examples

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

### Large Corpora 

When building a large retrieval corpus (>500M documents or >100B tokens), it is often necessary to shard the corpus and parallelize the encoding process.

## TODOs

- [] Save passage text instead of loading from file

## Contact

Please email me at `hyen@cs.princeton.edu` if you run into any issues or have any questions.
