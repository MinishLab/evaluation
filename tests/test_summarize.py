from pathlib import Path

from mteb.encoder_interface import Encoder

from evaluation import CustomMTEB, TaskType, get_tasks
from evaluation.utils import load_results, parse_mteb_results, print_leaderboard, summarize_results


def test_summarize(mock_encoder: Encoder, tmp_path: Path) -> None:
    """Test the evaluation with the CustomMTEB class."""
    tasks = get_tasks([TaskType.WORDSIM])
    evaluation = CustomMTEB(tasks)
    results = evaluation.run(mock_encoder, eval_splits=["test"], output_folder=tmp_path)

    # Option 1: Parse the results into a custom ResultSet format
    parsed_results = parse_mteb_results(mteb_results=results, model_name=mock_encoder.mteb_model_meta.name)
    task_scores = summarize_results(parsed_results)

    # Option 2: Load all results from the output folder
    results = load_results(tmp_path)
    task_scores = summarize_results(results)

    # Option 3: load a specific folder
    result_folder = tmp_path / mock_encoder.mteb_model_meta.name / mock_encoder.mteb_model_meta.revision
    results = load_results(result_folder)
    task_scores = summarize_results(results)

    # results = load_results("results/average_word_embeddings_komninos/sentence-transformers__average_word_embeddings_komninos")
    results = load_results("results/")
    # print(results)
    task_scores = summarize_results(results)
    print_leaderboard(task_scores)

    # print(task_scores["STS"])
    # print(task_scores.keys())
    # print(task_scores["WordSim"]["average_word_embeddings_komninos"])
    # print(task_scores["STS"]["average_word_embeddings_komninos"])

    # print(excluded_tasks)
    # print(excluded_task_names)
    # Check if they are the same
    # print(results)

    # print(parsed_results)
    # print(results[0])
    # print(type(results[0]))
    # # Assert that the results folder contains the results for all tasks
    # task_names = [task.metadata.name for task in tasks]
    # result_folder = tmp_path / mock_encoder.mteb_model_meta.name / mock_encoder.mteb_model_meta.revision
    # results = load_results(result_folder)
    # print(results)
    # task_scores = summarize_results(results)
    # print(task_scores)

    # results= load_results("results")
    # #print(results)
