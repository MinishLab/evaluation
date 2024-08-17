from pathlib import Path
from typing import TypeAlias

from model2vec.model.encoders import StaticEmbedder
from mteb.abstasks import AbsTask
from mteb.evaluation import MTEB
from sentence_transformers import SentenceTransformer

Embedder: TypeAlias = StaticEmbedder | SentenceTransformer
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mteb
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, metadata_load
from model2vec.logging_config import setup_logging
from mteb import MTEB_MAIN_EN, get_task

from evaluation.utilities import Embedder

_FORBIDDEN_JSONS = ("model_meta.json", "word_sim_benchmarks.json", "pearl_benchmark.json")


setup_logging()

logger = logging.getLogger(__name__)


class CustomMTEB(MTEB):
    def select_tasks(self, **kwargs: Any) -> None:
        """Override select_tasks to directly use passed task instances."""
        if self._tasks is not None:
            # Use tasks directly without reinitializing
            self.tasks = [task for task in self._tasks if isinstance(task, AbsTask)]
            # Initialize tasks_cls with the classes of the provided tasks
            self.tasks_cls = [type(task) for task in self.tasks]
            if len(self.tasks) != len(self._tasks):
                task_names = [task.metadata_dict["name"] for task in self.tasks]
                logger.warning(f"Some tasks may not have been initialized correctly: {task_names}")
            return

        # If no tasks are passed, fall back to the original behavior
        super().select_tasks(**kwargs)

    @property
    def available_task_types(self) -> set[str]:
        """Override to ensure task types are gathered from the instances."""
        return {task.metadata_dict["type"] for task in self.tasks}


def load_embedder(model_path: str, input_level: bool, word_level: bool, device: str) -> tuple[Embedder, str]:
    """
    Load the embedder.

    :param model_path: The path to the model.
    :param input_level: Whether to use input level embeddings.
    :param word_level: Whether to use word level embeddings.
    :param device: The device to use.
    :return: The embedder and the name of the model.
    :raises ValueError: If both input and word level are passed.
    """
    embedder: Embedder

    if input_level and word_level:
        raise ValueError("Both input and word level were passed.")

    if input_level:
        embedder = StaticEmbedder.from_model(model_path)
        name = embedder.name
    elif word_level:
        embedder = StaticEmbedder.from_vectors(model_path, apply_pca=True, apply_zipf=True)
        name = embedder.name
    else:
        # Always load on CPU
        embedder = SentenceTransformer(model_name_or_path=model_path, device="cpu")
        embedder = embedder.eval().to(device)
        model_name = Path(model_path).name.replace("_", "-")
        name = f"sentencetransformer_{model_name}"

    return embedder, name


@dataclass
class DatasetResult:
    """
    Scores for a single dataset.

    Attributes
    ----------
        scores: The scores for the dataset.
        time: The time it took to evaluate the dataset.

    """

    scores: list[float]
    time: float

    def mean(self) -> float:
        """Calculate the mean of all scores."""
        return float(np.mean(self.scores))


@dataclass
class ResultSet:
    """A set of results over multiple datasets."""

    datasets: dict[str, DatasetResult] = field(default_factory=dict)

    def summarize(self, task_subset: str | None = None) -> pd.Series:
        """Summarize the results by taking the mean of all datasets."""
        if task_subset is None:
            return pd.Series({name: result.mean() for name, result in self.datasets.items()})

        result_dict = {}
        for name in self.datasets:
            task = mteb.get_task(name)
            if task.metadata.type == task_subset:
                result_dict[name] = self.datasets[name].mean()

        return pd.Series(result_dict)

    def times(self) -> dict[str, float]:
        """Return the evaluation times for all datasets."""
        return {name: result.time for name, result in self.datasets.items()}


def _process_result_data(data: dict[str, Any]) -> DatasetResult:
    """
    Process a single result JSON.

    :param data: The data to process.
    :return: The processed data.
    """
    scores = []
    for score in data["scores"]["test"]:
        scores.append(score["main_score"])

    return DatasetResult(scores=scores, time=data["evaluation_time"])


def get_results_model_folder(model_name_path: Path) -> ResultSet:
    """
    Get the results for a single model folder.

    :param model_name_path: The path to the model folder.
    :return: The results for the model folder.
    """
    json_paths = model_name_path.glob("**/*.json")

    result = ResultSet()
    for json_path in json_paths:
        if json_path.name in _FORBIDDEN_JSONS:
            continue
        data = json.load(open(json_path))

        result.datasets[json_path.stem] = _process_result_data(data)

    return result


def get_results_from_hub(model_name: str) -> ResultSet | None:
    """
    Get the results from the model hub.

    :param model_name: The name of the model on the model hub.
    :return: The results.
    """
    readme = hf_hub_download(model_name, filename="README.md")
    try:
        results: list[dict[str, Any]] = metadata_load(readme)["model-index"][0]["results"]
    except KeyError:
        return None

    dataset_results = {}
    for result in results:
        task_name: str = result["dataset"]["name"]
        if not task_name.startswith("MTEB "):
            continue
        # NOTE: we split on space to remove MTEB and any suffixes.
        _, task_name, *_ = task_name.split()

        if not task_name in MTEB_MAIN_EN.tasks:
            continue

        try:
            main_score = get_task(task_name).metadata.main_score
            if main_score.startswith("cosine_"):
                main_score = main_score.replace("cosine_", "cos_sim_")
            elif main_score == "ap":
                main_score = "max_ap"
        except KeyError:
            continue

        metrics = {x["type"]: x["value"] for x in result["metrics"]}
        try:
            score = metrics[main_score] / 100
        except KeyError:
            logger.info(f"No main score {model_name}, {task_name}, {main_score}, {metrics}")
            continue

        dataset_results[task_name] = DatasetResult(scores=[score], time=0.0)

    return ResultSet(datasets=dataset_results)
