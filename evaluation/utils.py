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
from mteb.evaluation.LangMapping import LANG_MAPPING
from mteb.load_results import MTEBResults
from rich.logging import RichHandler
from scipy.stats._stats_py import SignificanceResult

from evaluation import TaskType, get_tasks

_FORBIDDEN_JSON = "model_meta.json"
_SUPPORTED_LANGS = {"default", "en-en", "en"}.union(LANG_MAPPING["en"])

_TASK_LIST_CQA = {
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
}

logger = logging.getLogger(__name__)


def setup_task_mappings() -> tuple[dict[str, list[str]], list[str]]:
    """
    Setup the task mappings for the evaluation.

    :return: A dictionary mapping task types to task names and a list of custom task names.
    """
    # Get all tasks
    all_tasks = get_tasks()
    # Create a dictionary mapping task types to task names
    task_type_to_tasks_mapping = defaultdict(list)

    # Get all WordSim tasks
    wordsim_tasks = get_tasks([TaskType.WORDSIM])
    wordsim_task_names = [task.metadata.name for task in wordsim_tasks]

    # Get all PEARL tasks
    pearl_tasks = get_tasks([TaskType.PEARL])
    pearl_task_names = [task.metadata.name for task in pearl_tasks]

    # Get all custom task names
    custom_task_names = wordsim_task_names + pearl_task_names

    # Populate the dictionary
    for task in all_tasks:
        if task.metadata.name in wordsim_task_names:
            task_type_to_tasks_mapping["WordSim"].append(task.metadata.name)
        elif task.metadata.name in pearl_task_names:
            task_type_to_tasks_mapping["PEARL"].append(task.metadata.name)
        else:
            task_type_to_tasks_mapping[task.metadata.type].append(task.metadata.name)

    return task_type_to_tasks_mapping, custom_task_names


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
    scores = [score["main_score"] for score in data["scores"]["test"] if score["hf_subset"] in _SUPPORTED_LANGS]
    scores = [score[0] if isinstance(score, list) else score for score in scores]

    return DatasetResult(scores=scores, time=data["evaluation_time"])


def parse_mteb_results(mteb_results: list[MTEBResults], model_name: str) -> dict[str, ResultSet]:
    """Parse MTEBResults into a dictionary of ResultSet objects."""
    dataset_results = {}

    for result in mteb_results:
        task_name = result.task_name
        test_scores = result.scores.get("test", [])
        if not test_scores:
            continue

        main_score = [score["main_score"] for score in test_scores if score["hf_subset"] in _SUPPORTED_LANGS][0]

        # Check if the main score is a SignificanceResult. If so, extract the statistic
        if isinstance(main_score, SignificanceResult):
            main_score = main_score.statistic

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
        # dataset_scores = []
        dataset_scores = {}
        task_summaries = {}

        for task_type in task_types:
            # Summarize the results for the specific task type
            task_summary = result_set.summarize(task_type=task_type)
            if task_type == "Retrieval":
                # Retrieval task is a special case, as it has multiple datasets for CQA
                scores = {}
                scores_cqa = []
                for name, score in task_summary.items():
                    if name not in _TASK_LIST_CQA:
                        scores[name] = score
                    else:
                        scores_cqa.append(score)
                    scores["CQADupstack"] = np.mean(scores_cqa)
                task_summary = pd.Series(scores)
            # Get the expected datasets for this task type
            expected_datasets = _task_type_to_tasks_mapping[task_type]
            # Check if the model has results for all required datasets, or the Retrieval task
            if set(task_summary.index) == set(expected_datasets) or task_type == "Retrieval":
                task_summaries[task_type] = task_summary.mean()
                for dataset, score in task_summary.items():
                    dataset_scores[dataset] = score
            else:
                task_summaries[task_type] = np.nan
                logger.warning(f"Model {model_name} is missing results for some datasets in task type {task_type}.")

        # Store task means but also collect all individual dataset scores for macro averaging
        model_scores[model_name] = {
            "task_means": pd.Series(task_summaries),
            "dataset_scores": dataset_scores,  # Collecting all dataset scores for macro averaging
        }

    return model_scores


def make_leaderboard(model_scores: dict[str, dict]) -> pd.DataFrame:
    """Make the leaderboard with the mean scores for each task and compute macro scores."""
    # Extract task means and dataset scores
    task_means = {model: scores["task_means"] for model, scores in model_scores.items()}
    dataset_scores = {model: scores["dataset_scores"] for model, scores in model_scores.items()}

    # Convert the task_means dictionary to a DataFrame for task-wise averaging
    leaderboard = pd.DataFrame(task_means)

    # Calculate the overall macro score for each model (mean of all datasets across all tasks)
    leaderboard.loc["Average (All)"] = {
        model: np.mean(list(scores.values())) for model, scores in dataset_scores.items()
    }

    # Filter out the custom task names from dataset_scores
    mteb_dataset_scores = {
        model: {dataset: score for dataset, score in scores.items() if dataset not in _custom_task_names}
        for model, scores in dataset_scores.items()
    }

    # Calculate the overall mean for MTEB tasks (excluding custom task names)
    leaderboard.loc["Average (MTEB)"] = {
        model: np.mean(list(scores.values())) for model, scores in mteb_dataset_scores.items()
    }

    # Multiply all values by 100 and format to 2 decimal places
    leaderboard = leaderboard.applymap(lambda x: f"{x * 100:.2f}" if isinstance(x, (int, float)) else x)

    # Replace NaN values with "N/A"
    leaderboard = leaderboard.fillna("N/A")

    # Transpose the DataFrame so models are in rows and task types in columns
    leaderboard = leaderboard.transpose().reset_index()

    # Rename the index column to "Model"
    leaderboard.rename(columns={"index": "Model"}, inplace=True)

    # Reorder columns to place "Average (All)" and "Average (MTEB)" right after "Model"
    columns = ["Model", "Average (All)", "Average (MTEB)"] + [
        col for col in leaderboard.columns if col not in ["Model", "Average (All)", "Average (MTEB)"]
    ]
    leaderboard = leaderboard[columns]

    return leaderboard
