# RetrievalTools

This repository contains a collection of useful retrieval tools, inspired by previous codebases, such as [Contriever](https://github.com/facebookresearch/contriever).
The main difference from previous repos like Contriever and DPR-Scale is the additional support for different types of retrievers like BM25, SentenceTransformers, and API calls. 
We also minimize the amount of extra scaffolding, such as Hydra configurations, to make the codebase more accessible to newcomers.
The goal of this repository is to provide a simple, easy-to-use, and efficient codebase for retrieval tasks.

We support the following operations:
 - Building dense indices with sharding
 - Retrieving across sharded dense indices on GPUs and merging the results efficiently
 - Common API for different retrieval models (e.g., BM25, SentenceTransformers, API calls)
 - Retrieval-from-context with different chunking stratgies

The code is especially customized towards a slurm-based cluster, and we optimize for easy paralleization and efficient memory usage.
This can be useful for large-scale retrieval tasks, where we might want to annotate a pre-training scale corpus with retrieval results (as was done in our previous paper [CEPE](https://arxiv.org/abs/2402.16617)).
We also provide a simple API for playing around with different retrieval models by enabling serving indices.

## Installation

To install the package, you should set up a conda environment and install PyTorch and FAISS (guide [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)).
Additionally, you should install `transformers` and `sentence-transformers`.
You also want to install `datatools` [here](https://github.com/CodeCreator/datatools).

## Usage

### Small Corpora

For smaller corpora (e.g., Wikipedia which has ~20M passages), we do not need to do anything super complicated.

### Large Corpora 

When building a large retrieval corpus (>100M documents or >100B tokens), it is often necessary to shard the corpus and parallelize the encoding process.

## Contact

Please email me at `hyen@cs.princeton.edu` if you run into any issues or have any questions.

<!-- ### A personal note

In my research journey, I have worked on a number of projects that involves retrieval.
From playing around with DensePhrases in the Sophomore year of college to my subsequent papers---MoQA, ALCE, CEPE, BRIGHT, and HELMET---nearly all of my projects have involved retrieval in some form.
Throughout this time, I have gotten familiar with many existing tools, and yet I often end up rewriting the same code over and over again.
Thus, I finally decided to create this repository to consolidate all of my retrieval tools in one place.

This repository is also inspired by my good friend Alex Wettig, who developed [datatools](https://github.com/CodeCreator/datatools), a useful collection of data processing tools. -->
<!-- Obviously this repo is named after his.  -->