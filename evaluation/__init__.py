from importlib.metadata import version

from evaluation.evaluation import CustomMTEB, TaskType, get_tasks

__all__ = ["CustomMTEB", "TaskType", "get_tasks"]
__version__ = version("evaluation")  # fetch version from install metadata
