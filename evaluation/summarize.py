from pathlib import Path

import pandas as pd
from mteb.load_results import MTEBResults

from evaluation.utils import DatasetResult, ResultSet, get_results_model_folder


def load_results(results_dir: str | Path, baseline_dir: str | Path | None = None) -> dict[str, ResultSet]:
    """
    Load results from the specified directory.

    :param results_dir: The root directory containing results for all models or a specific model directory.
    :param baseline_dir: The baseline directory, if any.
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

    if baseline_dir:
        results["baseline"] = get_results_model_folder(Path(baseline_dir))

    return results


def summarize_results(
    results: dict[str, ResultSet],
    task_subsets: list[str] | None = [
        "Classification",
        "Clustering",
        "PairClassification",
        "Reranking",
        "STS",
        "Summarization",
        "WordSim",
        "PEARL",
    ],
) -> dict[str, pd.DataFrame]:
    """Summarize the results by task subset, optionally comparing against a baseline."""
    summaries = {}
    if "baseline" in results:
        baseline_summary = results["baseline"].summarize()
        for model_name, result_set in results.items():
            if model_name != "baseline":
                summaries[model_name] = result_set.summarize() - baseline_summary
    else:
        summaries = {model_name: result_set.summarize() for model_name, result_set in results.items()}
    if task_subsets:
        task_scores = {}
        for task_subset in task_subsets:
            task_scores[task_subset] = summarize_task_subset(results, task_subset)

        return task_scores

    return summaries


def summarize_task_subset(results: dict[str, ResultSet], task_subset: str) -> pd.DataFrame:
    """Summarize the results for a specific task subset, assuming filtering is already done."""
    if "baseline" in results:
        baseline_summary = results["baseline"].summarize(task_subset=task_subset)
        return pd.DataFrame(
            {
                model_name: result_set.summarize(task_subset=task_subset) - baseline_summary
                for model_name, result_set in results.items()
                if model_name != "baseline"
            }
        )
    return pd.DataFrame(
        {model_name: result_set.summarize(task_subset=task_subset) for model_name, result_set in results.items()}
    )


def parse_mteb_results(mteb_results: list[MTEBResults], model_name: str) -> dict[str, ResultSet]:
    """Parse MTEBResults into a dictionary with the model name as the key."""
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
