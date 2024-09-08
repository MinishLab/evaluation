from pathlib import Path

from mteb.encoder_interface import Encoder

from evaluation import (
    CustomMTEB,
    TaskType,
    get_tasks,
    load_results,
    make_leaderboard,
    parse_mteb_results,
    summarize_results,
)


def test_summarize(mock_encoder: Encoder, tmp_path: Path) -> None:
    """Test the summarization of the evaluation results."""
    task_types = [task.value for task in TaskType]

    # Get the specified tasks and results
    tasks = get_tasks([TaskType.WORDSIM])
    evaluation = CustomMTEB(tasks)
    results = evaluation.run(mock_encoder, eval_splits=["test"], output_folder=tmp_path)

    # Set the model name
    model_name = f"{mock_encoder.mteb_model_meta.name}_{mock_encoder.mteb_model_meta.revision}"

    # Test option 1: Parse the results into a custom ResultSet format
    parsed_results = parse_mteb_results(mteb_results=results, model_name=model_name)
    model_scores = summarize_results(parsed_results)
    # Assert that all the task_types exist as keys in the model_scores
    assert all(task in model_scores[model_name]["task_means"].keys() for task in task_types)
    # Assert that every task_type has the mock_encoder name as a key
    assert model_name in model_scores
    # Ensure that print_leaderboard works
    make_leaderboard(model_scores)

    # Test option 2: Load all results from the output folder
    results = load_results(tmp_path)
    model_scores = summarize_results(results)
    # Assert that all the task_types exist as keys in the model_scores
    assert all(task in model_scores[model_name]["task_means"].keys() for task in task_types)
    # Assert that every task_type has the mock_encoder name as a key
    assert model_name in model_scores
    # Ensure that print_leaderboard works
    make_leaderboard(model_scores)

    # Test option 3: load a specific folder
    result_folder = tmp_path / mock_encoder.mteb_model_meta.name / mock_encoder.mteb_model_meta.revision
    results = load_results(result_folder)
    model_scores = summarize_results(results)
    # Assert that all the task_types exist as keys in the model_scores
    assert all(task in model_scores[model_name]["task_means"].keys() for task in task_types)
    # Assert that every task_type has the mock_encoder name as a key
    assert model_name in model_scores
    # Ensure that print_leaderboard works
    make_leaderboard(model_scores)
