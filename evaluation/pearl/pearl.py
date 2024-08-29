from __future__ import annotations

from typing import Any, Literal, cast

from datasets import DatasetDict, load_dataset
from mteb import TaskMetadata
from mteb.abstasks import AbsTask
from mteb.encoder_interface import Encoder

from evaluation.pearl.eval import eval_autofj, eval_bird, eval_clustering, eval_ppdb, eval_retrieval, eval_turney


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
            dataset = load_dataset("Lihuchen/pearl_benchmark", "umls", split="umls")
        else:
            dataset = load_dataset("Lihuchen/pearl_benchmark", self.dataset_name, split="test")
        self.dataset = DatasetDict(
            {
                "test": dataset,
            }
        )

    def evaluate(
        self, model: Encoder, split: str = "test", output_folder: str | None = None, **kwargs: Any
    ) -> dict[str, dict[str, float]]:
        """Evaluate the given model on the specified dataset split."""
        dataset_split = self.dataset[split]
        result = self._evaluate_subset(model, dataset_split)

        return {"default": {"accuracy": result, "main_score": result}}

    def _evaluate_subset(self, model: Encoder, dataset_split: str, **kwargs: Any) -> float:
        """Evaluate the given model on the specified dataset split."""
        match self.dataset_name:
            case "bird":
                return eval_bird(model, dataset_split)
            case "turney":
                return eval_turney(model, dataset_split)
            case "ppdb" | "ppdb_filtered":
                return eval_ppdb(model, dataset_split)
            case "yago" | "umls":
                kb_dataset = load_dataset("Lihuchen/pearl_benchmark", "kb", split=self.dataset_name)
                return eval_retrieval(model, kb_dataset, dataset_split)
            case "autofj":
                return eval_autofj(model, dataset_split)
            case "conll" | "bc5cdr":
                return eval_clustering(model, dataset_split, name=cast(Literal["conll", "bc5cdr"], self.dataset_name))
            case _:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")

    @classmethod
    def get_subtasks(cls) -> list[PEARL]:
        """Return a list of subtasks, one for each dataset in the PEARL benchmark."""
        return [cls(dataset_name=name) for name in cls.DATASET_TASK_MAPPING.keys()]
