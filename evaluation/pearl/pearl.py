from __future__ import annotations

from typing import Any, Literal, cast

from datasets import DatasetDict, load_dataset
from mteb import TaskMetadata
from mteb.abstasks import AbsTask

from evaluation.pearl.eval import eval_autofj, eval_bird, eval_clustering, eval_ppdb, eval_retrieval, eval_turney
from evaluation.utilities import Embedder


class PEARL(AbsTask):
    DATASET_TASK_MAPPING = {
        "bird": "Classification",
        "turney": "Classification",
        "ppdb": "Classification",
        "ppdb_filtered": "Classification",
        "yago": "Retrieval",
        "umls": "Retrieval",
        "autofj": "Retrieval",
        "conll": "Clustering",
        "bc5cdr": "Clustering",
    }

    def __init__(self, dataset_name: str, hf_subsets: Any = None, **kwargs: Any) -> None:
        """
        Initialize a PEARL task with the given dataset name.

        :param dataset_name: The name of the dataset to use.
        :param hf_subsets: The Hugging Face dataset splits to use.
        :param **kwargs: Additional keyword arguments.
        :raises ValueError: If the dataset name is unknown.
        """
        # Use the mapping to get the task type
        try:
            task_type = self.DATASET_TASK_MAPPING[dataset_name]
        except KeyError:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        self.dataset_name = dataset_name
        self.metadata = TaskMetadata(
            name=dataset_name,
            description=f"PEARL Task: {dataset_name}",
            dataset={
                "path": "Lihuchen/pearl_benchmark",
                "revision": "1.0.0",
            },
            reference=None,
            type=task_type,
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["en"],
            main_score="accuracy",
        )

        # Initialize the parent class after setting the metadata
        super().__init__(hf_subsets=hf_subsets, **kwargs)

    def load_data(self, eval_splits: Any = None) -> None:
        """Load the appropriate dataset based on the task name."""
        if self.dataset_name == "umls":
            dataset = load_dataset("Lihuchen/pearl_benchmark", "kb", split="umls")
        else:
            dataset = load_dataset("Lihuchen/pearl_benchmark", self.dataset_name, split="test")
        self.dataset = DatasetDict(
            {
                "test": dataset,
            }
        )

    def evaluate(
        self, model: Embedder, split: str = "test", output_folder: str | None = None, **kwargs: Any
    ) -> dict[str, dict[str, float]]:
        """Evaluate the given model on the specified dataset split."""
        dataset_split = self.dataset[split]
        result = self._evaluate_subset(model, dataset_split)
        return {"default": {"accuracy": result, "main_score": result}}

    def _evaluate_subset(self, model: Embedder, dataset_split: str, **kwargs: Any) -> float:
        """Evaluate the given model on the specified dataset split."""
        if self.dataset_name == "bird":
            return eval_bird(model, dataset_split)
        elif self.dataset_name == "turney":
            return eval_turney(model, dataset_split)
        elif self.dataset_name in ["ppdb", "ppdb_filtered"]:
            return eval_ppdb(model, dataset_split)
        elif self.dataset_name in ["yago", "umls"]:
            kb_dataset = load_dataset("Lihuchen/pearl_benchmark", "kb", split=self.dataset_name)
            return eval_retrieval(model, kb_dataset, dataset_split)
        elif self.dataset_name == "autofj":
            return eval_autofj(model, dataset_split)
        elif self.dataset_name in ["conll", "bc5cdr"]:
            return eval_clustering(model, dataset_split, name=cast(Literal["conll", "bc5cdr"], self.dataset_name))
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    @classmethod
    def get_subtasks(cls) -> list[PEARL]:
        """Return a list of subtasks, one for each dataset in the PEARL benchmark."""
        return [cls(dataset_name=name) for name in cls.DATASET_TASK_MAPPING.keys()]
