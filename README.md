# Evaluation

This repository can be used to evaluate word embeddings on several tasks. All tasks are implemented as [MTEB](https://github.com/embeddings-benchmark/mteb) tasks and can be run using the same interface.

## Usage

To run the evaluation on all available tasks, the following code can be used:

```python
from evaluation import CustomMTEB, get_tasks
from sentence_transformers import SentenceTransformer

# Define the model name
model_name = "average_word_embeddings_komninos"

# Get all available tasks
tasks = get_tasks()
evaluation = CustomMTEB(tasks=tasks)
model = SentenceTransformer(model_name)
results = evaluation.run(model, eval_splits=["test"], output_folder=f"results/{model_name}")
```

Alternatively, the evaluation can be run on a subset of tasks by specifying the task types:

```python
from evaluation import CustomMTEB, get_tasks, TaskType
from sentence_transformers import SentenceTransformer

# Define the model name
model_name = "average_word_embeddings_komninos"

# Get the specified tasks, in this case the classification and wordsim tasks
task_types = [TaskType.CLASSIFICATION, TaskType.WORDSIM]
tasks = get_tasks(task_types=task_types)

evaluation = CustomMTEB(tasks=tasks)
model = SentenceTransformer(model_name)
results = evaluation.run(model, eval_splits=["test"], output_folder=f"results/{model_name}")
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

Custom embedders can be used by implementing the [Encoder protocol](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/encoder_interface.py#L12) from `MTEB`.



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
All tasks from the [PEARL](https://arxiv.org/pdf/2401.10407) benchmark are supported:
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
