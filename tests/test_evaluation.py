from pathlib import Path

import pytest
from mteb.encoder_interface import Encoder

from evaluation import CustomMTEB, TaskType, get_tasks


def test_evaluation(mock_encoder: Encoder, tmp_path: Path) -> None:
    """Test the evaluation with the CustomMTEB class."""
    tasks = get_tasks([TaskType.WORDSIM])
    evaluation = CustomMTEB(tasks)
    results = evaluation.run(mock_encoder, eval_splits=["test"], output_folder=tmp_path)

    # Assert that the number of tasks and results are the same and that the results folder exists
    assert len(tasks) == len(results), "The number of tasks and results should be the same."
    assert (tmp_path).exists(), "The results folder should exist."

    # Assert that the results folder contains the results for all tasks
    task_names = [task.metadata.name for task in tasks]
    result_folder = tmp_path / mock_encoder.mteb_model_meta.name / mock_encoder.mteb_model_meta.revision

    assert all(
        (result_folder / f"{task_name}.json").exists() for task_name in task_names
    ), "All result files for the specified tasks should exist."

    # Ensure that get_tasks without any arguments works
    get_tasks()

    # Ensure that get_tasks with a string works
    get_tasks(["WordSim"])

    # Ensure that get_tasks with a non-existent task name raises an error
    with pytest.raises(ValueError):
        get_tasks(["non_existent_task"])
