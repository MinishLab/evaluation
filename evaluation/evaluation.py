import logging
from enum import Enum
from typing import Any

import mteb
from mteb.abstasks import AbsTask
from mteb.evaluation import MTEB

from evaluation.pearl.pearl import PEARL
from evaluation.wordsim.wordsim import WordSim

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enum for the different supported task types."""

    CLASSIFICATION = "Classification"
    CLUSTERING = "Clustering"
    PAIRCLASSIFICATION = "PairClassification"
    RERANKING = "Reranking"
    STS = "STS"
    SUMMARIZATION = "Summarization"
    PEARL = "PEARL"
    WORDSIM = "WordSim"


class CustomMTEB(MTEB):
    def select_tasks(self, *args: Any, **kwargs: Any) -> None:
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
        super().select_tasks(*args, **kwargs)

    @property
    def available_task_types(self) -> set[str]:
        """Override to ensure task types are gathered from the instances."""
        return {task.metadata_dict["type"] for task in self.tasks}


def get_tasks(task_types: list[TaskType] | None = None) -> list[AbsTask]:
    """
    Get the MTEB tasks that match the provided task types.

    :param task_types: The task types to include. If None, all task types are included.
    :return: The MTEB tasks that match the provided task types.
    :raises ValueError: If any task types are invalid.
    """
    # If no task types are provided, default to all task types
    if task_types is None:
        task_types = list(TaskType)
    else:
        # Validate that all items in task_types are instances of TaskType
        invalid_types = [task for task in task_types if not isinstance(task, TaskType)]
        if invalid_types:
            supported_types = ", ".join([t.name for t in TaskType])
            raise ValueError(
                f"Invalid task types: {invalid_types}. "
                f"All task types must be instances of TaskType. "
                f"Supported task types are: {supported_types}"
            )

    # Get the MTEB tasks that match the provided task types
    task_names = [
        task
        for task in mteb.MTEB_MAIN_EN.tasks
        if mteb.get_task(task).metadata.type in [task_type.value for task_type in task_types]
    ]
    tasks = [mteb.get_task(task) for task in task_names]

    # If WordSim is in the task types, add the WordSim subtasks
    if TaskType.WORDSIM in task_types:
        wordsim_subtasks = WordSim.get_subtasks()
        tasks.extend(wordsim_subtasks)

    # If PEARL is in the task types, add the PEARL subtasks
    if TaskType.PEARL in task_types:
        pearl_subtasks = PEARL.get_subtasks()
        tasks.extend(pearl_subtasks)

    return tasks
