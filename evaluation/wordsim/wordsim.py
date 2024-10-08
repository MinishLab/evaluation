from __future__ import annotations

from importlib import resources
from typing import Any

import datasets
from mteb import TaskMetadata
from mteb.abstasks import AbsTaskSTS

from evaluation.wordsim.tasks import wordsim_tasks


class WordSim(AbsTaskSTS):
    def __init__(self, dataset_name: str | None = None, hf_subsets: Any = None, **kwargs: Any) -> None:
        """
        Initialize a WordSim task with the given dataset name.

        :param dataset_name: The name of the dataset to use.
        :param hf_subsets: The Hugging Face dataset splits to use.
        :param **kwargs: Additional keyword arguments.
        """
        super().__init__(hf_subsets=hf_subsets, **kwargs)
        self.dataset_name = dataset_name
        self.metadata = TaskMetadata(
            name=dataset_name if dataset_name else "WordSim",
            description=f"Custom Word Similarity Task: {dataset_name}"
            if dataset_name
            else "Custom Word Similarity Task with Multiple Datasets.",
            reference=None,
            type="STS",
            category="s2s",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["en"],
            main_score="spearman",
            dataset={
                "path": "evaluation/wordsim/tasks.py",
                "revision": "1.0.0",
            },
        )
        self.dataset_splits: dict[str, dict] = {}

    @property
    def min_score(self) -> int:
        """Minimum score for the similarity task."""
        return -1

    @property
    def max_score(self) -> int:
        """Maximum score for the similarity task."""
        return 1

    def load_data(self, eval_splits: Any = None) -> None:
        """Load the WordSim datasets."""
        # Load the data for each task
        for task in wordsim_tasks:
            sentence1 = []
            sentence2 = []
            scores = []

            index1 = task.index1
            index2 = task.index2
            target = task.target

            with resources.open_text("evaluation.wordsim.data", task.file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    # Remove underscores from the words
                    parts = [part.replace("_", " ") for part in parts]
                    word1 = parts[index1]
                    word2 = parts[index2]

                    similarity = float(parts[target])

                    sentence1.append(word1)
                    sentence2.append(word2)
                    scores.append(similarity)

            dataset_name = task.task
            self.dataset_splits[dataset_name] = datasets.Dataset.from_dict(
                {
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "score": scores,
                }
            )
        if self.dataset_name:
            self.dataset = datasets.DatasetDict(
                {
                    "test": self.dataset_splits[self.dataset_name],
                }
            )
        else:
            self.dataset = datasets.DatasetDict(self.dataset_splits)

    @classmethod
    def get_subtasks(cls) -> list[WordSim]:
        """Return a list of subtasks, one for each dataset."""
        instance = cls()
        instance.load_data()
        return [cls(dataset_name=name) for name in instance.dataset_splits.keys()]
