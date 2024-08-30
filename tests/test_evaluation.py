from pathlib import Path
from typing import Any, Sequence
from unittest.mock import create_autospec

import pytest
import torch
from mteb.encoder_interface import Encoder

from evaluation import CustomMTEB, TaskType, get_tasks


@pytest.fixture
def mock_encoder() -> Encoder:
    """Return a mock encoder that follows the Encoder protocol."""
    mock_encoder = create_autospec(Encoder, instance=True)

    # Mock the encode method
    def mock_encode(sentences: Sequence[str], prompt_name: str | None = None, **kwargs: Any) -> torch.Tensor:
        """Return random embeddings for the sentence."""
        return torch.rand(len(sentences), 300)

    # Set the side effect of the mock
    mock_encoder.encode.side_effect = mock_encode

    return mock_encoder


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
    result_folder = tmp_path / "no_model_name_available" / "no_revision_available"
    assert all(
        (result_folder / f"{task_name}.json").exists() for task_name in task_names
    ), "All result files for the specified tasks should exist."

    # Ensure that get_tasks without any arguments works
    get_tasks()

    # Ensure that loading tasks with a string works
    get_tasks(["WordSim"])

    # Ensure that loading tasks with a non-existent task name raises an error
    with pytest.raises(ValueError):
        get_tasks(["non_existent_task"])
