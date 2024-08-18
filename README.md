# Evaluation

This repository can be used to evaluate word embeddings on several tasks. All tasks are implemented as MTEB tasks and can be run using the same interface.

## Usage

To evaluate a model, run the following command:

```bash
python evaluation --model-name <model>
```

The model can either be a path to a Huggingface sentence transformer, or a path to REACH embeddings.

## Tasks

The following tasks are supported:

### MTEB
All tasks in [MTEB](https://github.com/embeddings-benchmark/mteb) are supported:
- Classification
- Clustering
- PairClassification
- Reranking
- Retrieval
- STS
- Summarization

### WordSim
- A collection of single word similarity tasks

### PEARL
The [PEARL](https://arxiv.org/pdf/2401.10407) benchmark:
- Paraphrase Classification
- Phrase Similarity
- Entity Retrieval
- Entity Clustering
- Fuzzy Join
