# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from argparse import ArgumentParser

import mteb
from model2vec.logging_config import setup_logging

from evaluation.pearl.pearl import PEARL
from evaluation.utilities import CustomMTEB, load_embedder
from evaluation.wordsim.wordsim import WordSim

logger = logging.getLogger(__name__)


# NOTE: we leave out "Retrieval" because it is too expensive to run.
ALL_TASKS_TYPES = (
    "Classification",
    "Clustering",
    "PairClassification",
    "Reranking",
    "STS",
    "Summarization",
    "PEARL",
    "WordSim",
)


def main() -> None:
    """Main function for evaluating the MTEB benchmark and several custom tasks."""
    parser = ArgumentParser()
    parser.add_argument("--model-path", help="The model to use.", required=True)
    parser.add_argument("--input", action="store_true")
    parser.add_argument("--word-level", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--task-types", nargs="+", default=ALL_TASKS_TYPES)
    args = parser.parse_args()
    embedder, name = load_embedder(args.model_path, args.input, args.word_level, args.device)

    # Check that the task types are valid
    for task_type in args.task_types:
        if task_type not in ALL_TASKS_TYPES:
            raise ValueError(f"Invalid task type: {task_type}")

    # If a suffix is provided, add it to the name
    if args.suffix:
        name = f"{name}_{args.suffix}"

    # Get the MTEB tasks that match the provided task types
    task_names = [task for task in mteb.MTEB_MAIN_EN.tasks if mteb.get_task(task).metadata.type in args.task_types]
    tasks = [mteb.get_task(task) for task in task_names]

    # If WordSim is in the task types, add the WordSim subtasks
    if "WordSim" in args.task_types:
        wordsim_subtasks = WordSim.get_subtasks()
        tasks.extend(wordsim_subtasks)

    # If PEARL is in the task types, add the PEARL subtasks
    if "PEARL" in args.task_types:
        pearl_subtasks = PEARL.get_subtasks()
        tasks.extend(pearl_subtasks)

    evaluation = CustomMTEB(tasks=tasks)
    evaluation.run(embedder, eval_splits=["test"], output_folder=f"results/{name}")


if __name__ == "__main__":
    setup_logging()

    main()
