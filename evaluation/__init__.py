from importlib.metadata import version

from evaluation.evaluation import CustomMTEB, TaskType, get_tasks
from evaluation.utils import load_results, parse_mteb_results, print_leaderboard, summarize_results

__all__ = [
    "CustomMTEB",
    "TaskType",
    "get_tasks",
    "load_results",
    "parse_mteb_results",
    "print_leaderboard",
    "summarize_results",
]
__version__ = version("evaluation")  # fetch version from install metadata
