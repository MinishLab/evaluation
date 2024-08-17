from typing import Any

from datasets import DatasetDict, load_dataset
from mteb import TaskMetadata
from mteb.abstasks import AbsTask

from evaluation.pearl.eval import eval_autofj, eval_bird, eval_clustering, eval_ppdb, eval_retrieval, eval_turney
from evaluation.utilities import Embedder


class PEARL(AbsTask):
    def __init__(self, dataset_name: str = None, hf_subsets: Any = None, **kwargs: Any) -> None:
        """
        Initialize a PEARL task with the given dataset name.

        :param dataset_name: The name of the dataset to use.
        :param hf_subsets: The Hugging Face dataset splits to use.
        :param **kwargs: Additional keyword arguments.
        :raises ValueError: If the dataset name is unknown.
        """
        if dataset_name in ["bird", "turney", "ppdb", "ppdb_filtered"]:
            task_type = "Classification"
        elif dataset_name in ["yago", "umls", "autofj"]:
            task_type = "Retrieval"
        elif dataset_name in ["conll", "bc5cdr"]:
            task_type = "Clustering"
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        self.dataset_name = dataset_name
        self.metadata = TaskMetadata(
            name=dataset_name if dataset_name else "PEARL",
            description=f"PEARL Task: {dataset_name}" if dataset_name else "PEARL Benchmark with Multiple Datasets.",
            dataset={
                "path": "Lihuchen/pearl_benchmark",
                "revision": "1.0.0",
            },
            reference=None,
            type=task_type,
            modalities=["text"],
            eval_splits=["train", "test"],
            eval_langs=["en"],
            main_score="accuracy",
        )

        # Initialize the parent class after setting the metadata
        super().__init__(hf_subsets=hf_subsets, **kwargs)

    def load_data(self, eval_splits: Any = None) -> None:
        """Load the appropriate dataset based on the task name."""
        if self.dataset_name == "bird":
            dataset = load_dataset("Lihuchen/pearl_benchmark", "bird", split="test")
        elif self.dataset_name == "turney":
            dataset = load_dataset("Lihuchen/pearl_benchmark", "turney", split="test")
        elif self.dataset_name == "ppdb":
            dataset = load_dataset("Lihuchen/pearl_benchmark", "ppdb", split="test")
        elif self.dataset_name == "ppdb_filtered":
            dataset = load_dataset("Lihuchen/pearl_benchmark", "ppdb_filtered", split="test")
        elif self.dataset_name == "yago":
            dataset = load_dataset("Lihuchen/pearl_benchmark", "yago", split="test")
        elif self.dataset_name == "umls":
            dataset = load_dataset("Lihuchen/pearl_benchmark", "umls", split="umls")
        elif self.dataset_name == "conll":
            dataset = load_dataset("Lihuchen/pearl_benchmark", "conll", split="test")
        elif self.dataset_name == "bc5cdr":
            dataset = load_dataset("Lihuchen/pearl_benchmark", "bc5cdr", split="test")
        elif self.dataset_name == "autofj":
            dataset = load_dataset("Lihuchen/pearl_benchmark", "autofj", split="test")
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

        self.dataset = DatasetDict(
            {
                "test": dataset,
            }
        )

    def evaluate(
        self, model: Embedder, split: str = "test", output_folder: str = None, **kwargs: Any
    ) -> dict[str, dict[str, float]]:
        """Evaluate the given model on the specified dataset split."""
        dataset_split = self.dataset[split]
        result = self._evaluate_subset(model, dataset_split)
        return {"default": {"accuracy": result, "main_score": result}}

    def _evaluate_subset(self, model: Embedder, dataset_split: str, **kwargs: Any) -> float:
        """Evaluate the given model on the specified dataset split."""
        if self.metadata.type == "Classification":
            if self.dataset_name == "bird":
                return eval_bird(model, dataset_split)
            elif self.dataset_name == "turney":
                return eval_turney(model, dataset_split)
            elif self.dataset_name in ["ppdb", "ppdb_filtered"]:
                return eval_ppdb(model, dataset_split)
            else:
                raise ValueError(f"Unknown classification dataset: {self.dataset_name}")

        elif self.metadata.type == "Retrieval":
            if self.dataset_name in ["yago", "umls"]:
                kb_dataset = load_dataset("Lihuchen/pearl_benchmark", "kb", split=self.dataset_name)
                return eval_retrieval(model, kb_dataset, dataset_split)
            elif self.dataset_name == "autofj":
                return eval_autofj(model, dataset_split)
            else:
                raise ValueError(f"Unknown retrieval dataset: {self.dataset_name}")

        elif self.metadata.type == "Clustering":
            if self.dataset_name in ["conll", "bc5cdr"]:
                return eval_clustering(model, dataset_split, name=self.dataset_name)
            else:
                raise ValueError(f"Unknown clustering dataset: {self.dataset_name}")

        else:
            raise ValueError(f"Unknown task type: {self.metadata.type}")

    @classmethod
    def get_subtasks(cls) -> list["PEARL"]:
        """Return a list of subtasks, one for each dataset in the PEARL benchmark."""
        dataset_names = ["bird", "turney", "ppdb", "ppdb_filtered", "yago", "umls", "conll", "bc5cdr", "autofj"]
        return [cls(dataset_name=name) for name in dataset_names]
