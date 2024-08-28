from importlib.metadata import version

from evaluation.evaluation import CustomMTEB, TaskType, get_tasks  # noqa: F401

__version__ = version("evaluation")  # fetch version from install metadata
