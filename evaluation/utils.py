import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import click
import mteb
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, metadata_load
from mteb import MTEB_MAIN_EN, get_task
from mteb.encoder_interface import Encoder
from rich.logging import RichHandler
from sentence_transformers import SentenceTransformer

_FORBIDDEN_JSONS = ("model_meta.json", "word_sim_benchmarks.json", "pearl_benchmark.json")


logger = logging.getLogger(__name__)

Embedder: TypeAlias = Encoder | SentenceTransformer


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


def setup_logging() -> None:
    """Simple logging setup."""
    logging.basicConfig(
        level="INFO",
        format="%(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])],
    )


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
