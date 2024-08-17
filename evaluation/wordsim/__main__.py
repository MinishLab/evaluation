# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
from pathlib import Path

from model2vec.logging_config import setup_logging

from evaluation.utilities import get_default_argparser, load_embedder
from evaluation.wordsim.utilities import calculate_spearman_correlation, create_vocab_and_tasks_dict

logger = logging.getLogger(__name__)


def main() -> None:
    """Main function for evaluating the word similarity benchmarks."""
    parser = get_default_argparser()
    args = parser.parse_args()

    embedder, name = load_embedder(args.model_path, args.input, args.word_level, args.device)
    # Define the path the JSONL file containing paths and info about the tasks to evaluate
    tasks_file_path = "evaluation/wordsim/tasks.jsonl"

    # Read the JSONL data from the file into a list of tasks
    with open(tasks_file_path, "r") as file:
        tasks = [json.loads(line) for line in file]

    # Create the vocabulary and tasks dictionary
    vocab, tasks_dict = create_vocab_and_tasks_dict(tasks)

    task_scores = {}

    # Iterate over the tasks and calculate the Spearman correlation
    for task_name, task_data in tasks_dict.items():
        score, n_oov_trials = calculate_spearman_correlation(
            task=task_data,
            embedder=embedder,
        )
        task_scores[task_name] = score, n_oov_trials

    # Calculate the average score
    all_scores, _ = zip(*task_scores.values())
    average_score = round(sum(all_scores) / len(all_scores))
    task_scores["average"] = average_score

    if args.suffix:
        name = f"{name}_{args.suffix}"

    # Create the results directory if it does not exist
    Path(f"results/{name}").mkdir(parents=True, exist_ok=True)

    # Save the scores json to a file
    with open(f"results/{name}/word_sim_benchmarks.json", "w") as file:
        json.dump(task_scores, file, indent=4)

    logger.info(task_scores)
    logger.info(f"Results saved to results/{name}/wordsim_benchmarks.json")


if __name__ == "__main__":
    setup_logging()

    main()
