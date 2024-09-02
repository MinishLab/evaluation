import json
import logging
from collections import defaultdict
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

_FORBIDDEN_JSON = "model_meta.json"

logger = logging.getLogger(__name__)


def setup_task_mappings() -> tuple[dict[str, list[str]], list[str]]:
    """
    Setup the task mappings for the evaluation.

    :return: A dictionary mapping task types to task names and a list of custom task names.
    """
    # Get all tasks
    _all_tasks = get_tasks()
    # Create a dictionary mapping task types to task names
    _task_type_to_tasks_mapping = defaultdict(list)

    # Get all WordSim tasks
    _wordsim_tasks = get_tasks([TaskType.WORDSIM])
    _wordsim_task_names = [task.metadata.name for task in _wordsim_tasks]

    # Get all PEARL tasks
    _pearl_tasks = get_tasks([TaskType.PEARL])
    _pearl_task_names = [task.metadata.name for task in _pearl_tasks]

    # Get all custom task names
    _custom_task_names = _wordsim_task_names + _pearl_task_names

    # Populate the dictionary
    for task in _all_tasks:
        if task.metadata.name in _wordsim_task_names:
            _task_type_to_tasks_mapping["WordSim"].append(task.metadata.name)
        elif task.metadata.name in _pearl_task_names:
            _task_type_to_tasks_mapping["PEARL"].append(task.metadata.name)
        else:
            _task_type_to_tasks_mapping[task.metadata.type].append(task.metadata.name)

    return _task_type_to_tasks_mapping, _custom_task_names


_task_type_to_tasks_mapping, _custom_task_names = setup_task_mappings()


def setup_logging() -> None:
    """Simple logging setup."""
    logging.basicConfig(
        level="INFO",
        format="%(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])],
    )


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

    def summarize(self, task_type: str) -> pd.Series:
        """Summarize the results by taking the mean of all datasets."""
        result_dict = {}
        for name, result in self.datasets.items():
            # Check if the task is a custom task or an MTEB task
            if name not in _custom_task_names:
                task = mteb.get_task(name)
                if task.metadata.type == task_type:
                    result_dict[name] = result.mean()
            if task_type == "WordSim":
                if name in _task_type_to_tasks_mapping["WordSim"]:
                    result_dict[name] = result.mean()
            elif task_type == "PEARL":
                if name in _task_type_to_tasks_mapping["PEARL"]:
                    result_dict[name] = result.mean()

        return pd.Series(result_dict)

    def times(self) -> dict[str, float]:
        """Return the evaluation times for all datasets."""
        return {name: result.time for name, result in self.datasets.items()}


def load_results(results_dir: str | Path) -> dict[str, ResultSet]:
    """
    Load results from the specified directory.

    :param results_dir: The root directory containing results for all models.
    :return: A dictionary of model names to ResultSet objects.
    """
    results: defaultdict = defaultdict(ResultSet)
    results_path = Path(results_dir).resolve()

    # Glob for all JSON files in the directory and subdirectories
    json_paths = results_path.glob("**/*.json")

    for json_path in json_paths:
        # Construct the model name from the parent folder (model_revision) and its parent (model_name)
        model_revision = json_path.parent.name
        model_name = json_path.parent.parent.name
        if model_name == "no_model_name_available":
            logger.warning(f"Model name not available for {json_path}. Skipping.")
            continue
        elif model_revision == "no_revision_available":
            full_model_name = model_name
        else:
            full_model_name = f"{model_name}_{model_revision}"

        if json_path.name != _FORBIDDEN_JSON:
            with open(json_path) as f:
                data = json.load(f)

            results[full_model_name].datasets[json_path.stem] = _process_result_data(data)

    return dict(results)


def _process_result_data(data: dict[str, Any]) -> DatasetResult:
    """
    Process a single result JSON.

    :param data: The data to process.
    :return: The processed data.
    """
    scores = [score["main_score"] for score in data["scores"]["test"]]
    return DatasetResult(scores=scores, time=data["evaluation_time"])


def parse_mteb_results(mteb_results: list[MTEBResults], model_name: str) -> dict[str, ResultSet]:
    """Parse MTEBResults into a dictionary of ResultSet objects."""
    dataset_results = {}

    for result in mteb_results:
        task_name = result.task_name
        test_scores = result.scores.get("test", [])
        if not test_scores:
            continue

        main_score = test_scores[0]["main_score"]

        # Populate the DatasetResult
        dataset_results[task_name] = DatasetResult(scores=[main_score], time=result.evaluation_time)

    return {model_name: ResultSet(datasets=dataset_results)}


def summarize_results(
    results: dict[str, ResultSet],
) -> dict[str, pd.DataFrame]:
    """
    Summarize the results for all models and tasks.

    :param results: The results to summarize.
    :return: A dictionary mapping model names to DataFrames containing the mean scores for each task, if available.
    """
    model_scores = {}
    task_types = [task.value for task in TaskType]

    for model_name, result_set in results.items():
        model_summary = {}
        for task_type in task_types:
            # Summarize the results for the specific task type
            task_summary = result_set.summarize(task_type=task_type)

            # Get the expected datasets for this task type
            expected_datasets = _task_type_to_tasks_mapping[task_type]

            # Check if the model has results for all required datasets
            if set(task_summary.index) == set(expected_datasets):
                # All datasets are present, calculate the mean
                model_summary[task_type] = task_summary.mean()
            else:
                # Missing datasets, set mean to NaN
                model_summary[task_type] = np.nan
                logger.warning(f"Model {model_name} is missing results for some datasets in task type {task_type}.")

        # Convert model_summary to a pandas Series for easier handling later
        model_scores[model_name] = pd.Series(model_summary)

    # Convert the model_scores dictionary to a DataFrame
    return pd.DataFrame(model_scores)


def make_leaderboard(model_scores: dict[str, pd.Series]) -> pd.DataFrame:
    """Make the leaderboard with the mean scores for each task."""
    # Convert the model_scores dictionary to a DataFrame
    leaderboard = pd.DataFrame(model_scores)

    # Calculate the overall mean for all tasks (only if every task has a score)
    all_task_types = leaderboard.index.tolist()
    leaderboard.loc["Average (All)"] = leaderboard.apply(
        lambda row: row.mean() if not row.isna().any() else np.nan, axis=0
    )

    # Calculate the overall mean for MTEB tasks (excluding PEARL and WordSim)
    mteb_task_types = [task for task in all_task_types if task not in {"PEARL", "WordSim"}]
    leaderboard.loc["Average (MTEB)"] = leaderboard.loc[mteb_task_types].apply(
        lambda row: row.mean() if not row.isna().any() else np.nan, axis=0
    )

    # Replace NaN values with "N/A"
    leaderboard = leaderboard.fillna("N/A")

    # Transpose the DataFrame so models are in rows and task types in columns
    leaderboard = leaderboard.transpose().reset_index()

    # Rename the index column to "Model"
    leaderboard.rename(columns={"index": "Model"}, inplace=True)

    return leaderboard
