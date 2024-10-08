# Evaluation

This repository can be used to evaluate word embeddings on several tasks. All tasks are implemented as [MTEB](https://github.com/embeddings-benchmark/mteb) tasks and can be run using the same interface.

## Usage

To run the evaluation on all available tasks and summarize the results, the following code can be used:

```python
from sentence_transformers import SentenceTransformer

from evaluation import CustomMTEB, get_tasks, parse_mteb_results, make_leaderboard, summarize_results

# Define the model name
model_name = "average_word_embeddings_glove.6B.300d"

# Get all available tasks
tasks = get_tasks()
# Define the CustomMTEB object with the specified tasks
evaluation = CustomMTEB(tasks=tasks)
model = SentenceTransformer(model_name)
results = evaluation.run(model, eval_splits=["test"], output_folder=f"results/{model_name}")

# Parse the results and summarize them
parsed_results = parse_mteb_results(mteb_results=results, model_name=model_name)
task_scores = summarize_results(parsed_results)
# Print the results in a leaderboard format
print(make_leaderboard(task_scores))
```

This will print a markdown table similar to the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard), e.g.:

```
| Model            |   Average (All) |   Average (MTEB) |   Classification |   Clustering |   PairClassification |   Reranking |   Retrieval |    STS |   Summarization |   PEARL |   WordSim |
|:-----------------|----------------:|-----------------:|-----------------:|-------------:|---------------------:|------------:|------------:|-------:|----------------:|--------:|----------:|
| GloVe_300d       |           42.84 |            42.36 |            57.31 |        27.66 |                72.48 |       43.3  |       22.78 |  61.9  |           28.81 |   45.65 |     43.05 |
```

Alternatively, the evaluation can be run on a subset of tasks by specifying the task types:

```python
from evaluation import CustomMTEB, get_tasks, TaskType
from sentence_transformers import SentenceTransformer

# Define the model name
model_name = "average_word_embeddings_glove.6B.300d"

# Get the specified tasks, in this case the classification and wordsim tasks
task_types = [TaskType.CLASSIFICATION, TaskType.WORDSIM]
tasks = get_tasks(task_types=task_types)

# Define the CustomMTEB object with the specified tasks
evaluation = CustomMTEB(tasks=tasks)
# Run the rest of the evaluation and summarization as before
```

The following tasks are supported and can be used via the `TaskType` enum:
```python
- CLASSIFICATION
- CLUSTERING
- PAIRCLASSIFICATION
- RERANKING
- RETRIEVAL
- STS
- SUMMARIZATION
- WORDSIM
- PEARL
```
Alternatively, the task types can also be specified as a list of strings, e.g. `task_types=["Classification", "WordSim"]`.

Custom embedders can be used by implementing the [Encoder protocol](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/encoder_interface.py#L12) from `MTEB`.

### Summarizing results

The `summarize_results` function can be used to summarize results from an existing results folder, e.g.:

```python
from evaluation import load_results, make_leaderboard, summarize_results

# To summarize all models in a results folder:
results = load_results("results/")
task_scores = summarize_results(results)
print(make_leaderboard(task_scores))

# To summarize a single model:
results = load_results("results/average_word_embeddings_glove.6B.300d/")
task_scores = summarize_results(results)
print(make_leaderboard(task_scores))
```


## Supported Tasks
All tasks from [MTEB](https://github.com/embeddings-benchmark/mteb) are supported:
- Classification
- Clustering
- PairClassification
- Reranking
- Retrieval
- STS
- Summarization

### PEARL
All tasks from the [PEARL paper](https://arxiv.org/pdf/2401.10407) benchmark are supported (PEARL codebase [here](https://github.com/tigerchen52/PEARL)):
- Paraphrase Classification
- Phrase Similarity
- Entity Retrieval
- Entity Clustering
- Fuzzy Join

### WordSim
A collection of single word similarity datasets are supported:
- RareWord
- MEN
- SimLex
- rel353
- simverb
- muturk
- Card660
