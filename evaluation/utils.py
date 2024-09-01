import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import mteb
import numpy as np
import pandas as pd
from mteb.load_results import MTEBResults
from rich.logging import RichHandler

from evaluation import TaskType, get_tasks

_FORBIDDEN_JSONS = ("model_meta.json", "word_sim_benchmarks.json", "pearl_benchmark.json")
CUSTOM_TASKS = get_tasks(["WordSim", "PEARL"])
CUSTOM_TASK_TO_NAME_MAPPING = {task.metadata.name: task for task in CUSTOM_TASKS}

logger = logging.getLogger(__name__)


@dataclass
class DatasetResult:
    """
    Scores for a single dataset.

    Attributes
    ----------
        scores: The scores for the dataset.
        time: The time it took to evaluate the dataset.
        metrics: The metrics for the dataset.

    """

    scores: list[float]
    time: float
    metrics: dict[str, float] = field(default_factory=dict)

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
        for name, result in self.datasets.items():
            # Check if the task is a custom task or an MTEB task
            if name not in CUSTOM_TASK_TO_NAME_MAPPING:
                task = mteb.get_task(name)
                if task.metadata.type == task_subset:
                    result_dict[name] = result.mean()
            else:
                if task_subset in {"WordSim", "PEARL"}:
                    task = CUSTOM_TASK_TO_NAME_MAPPING[name]
                    result_dict[name] = result.mean()

        return pd.Series(result_dict)

    def times(self) -> dict[str, float]:
        """Return the evaluation times for all datasets."""
        return {name: result.time for name, result in self.datasets.items()}


def setup_logging() -> None:
    """Simple logging setup."""
    logging.basicConfig(
        level="INFO",
        format="%(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])],
    )


def load_results(
    results_dir: str | Path,
) -> dict[str, ResultSet]:
    """
    Load results from the specified directory.

    :param results_dir: The root directory containing results for all models or a specific model directory.
    :return: A dictionary of model names to ResultSet objects.
    """
    results = {}
    results_path = Path(results_dir)

    # Determine if results_dir is a specific model directory or contains multiple model directories
    if (results_path / "model_meta.json").exists() or any(results_path.glob("*.json")):
        # If the directory contains model result files directly, treat it as a specific model directory
        model_name = results_path.parent.name  # Use the parent directory's name as the model_name
        results[model_name] = get_results_model_folder(results_path)
    else:
        # Otherwise, treat it as a directory containing multiple model directories
        for model_name_path in results_path.iterdir():
            if model_name_path.is_dir():
                name = model_name_path.name
                results[name] = get_results_model_folder(model_name_path)

    return results


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


def parse_mteb_results(mteb_results: list[MTEBResults], model_name: str) -> dict[str, ResultSet]:
    """Parse MTEBResults into a dictionary of ResultSet objects."""
    dataset_results = {}

    for result in mteb_results:
        task_name = result.task_name
        test_scores = result.scores.get("test", [])
        if not test_scores:
            continue

        main_score = test_scores[0].get("main_score")
        metrics = {key: score for key, score in test_scores[0].items() if key != "hf_subset" and key != "languages"}

        # Populate the DatasetResult
        dataset_results[task_name] = DatasetResult(scores=[main_score], time=result.evaluation_time, metrics=metrics)

    return {model_name: ResultSet(datasets=dataset_results)}


def summarize_results(
    results: dict[str, ResultSet],
) -> dict[str, pd.DataFrame]:
    """Summarize the results by task subset."""
    task_types = [task.value for task in TaskType]
    task_scores = {}
    for task_subset in task_types:
        subset_summary = _summarize_task_subset(results, task_subset)
        task_scores[task_subset] = subset_summary
        # Calculate the mean for each model within the task subset
        for model_name in subset_summary.columns:
            task_scores[task_subset].loc["mean", model_name] = subset_summary[model_name].mean()

    return task_scores


def _summarize_task_subset(results: dict[str, ResultSet], task_subset: str) -> pd.DataFrame:
    """Summarize the results for a specific task subset."""
    return pd.DataFrame(
        {model_name: result_set.summarize(task_subset=task_subset) for model_name, result_set in results.items()}
    )


def print_leaderboard(task_scores: dict[str, pd.DataFrame]) -> None:
    """Print the leaderboard with the mean scores for each task using tabulate for better formatting."""
    # Extract the mean scores for each task subset and each model
    leaderboard = pd.DataFrame()

    for task_subset, scores in task_scores.items():
        leaderboard[task_subset] = scores.loc["mean"]

    # Calculate the overall mean
    leaderboard["Average"] = leaderboard.mean(axis=1)

    # Replace NaN values with "N/A"
    leaderboard = leaderboard.fillna("N/A")

    # Sort the leaderboard by the Average (ignoring N/A values)
    leaderboard = leaderboard.sort_values(by="Average", ascending=False)

    # Reset the index to make the model names a column
    leaderboard = leaderboard.reset_index()

    # Rename the index column to "Model"
    leaderboard.rename(columns={leaderboard.columns[0]: "Model"}, inplace=True)

    print(leaderboard.to_markdown(index=False))  # noqa: T201
