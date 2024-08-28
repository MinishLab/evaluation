from dataclasses import dataclass


@dataclass
class WordSimTask:
    """
    A WordSim task.

    Attributes
    ----------
        task: The name of the task.
        file: The file path to the dataset.
        index1: The index of the first word in the dataset.
        index2: The index of the second word in the dataset.
        target: The index of the target value in the dataset

    """

    task: str
    file: str
    index1: int
    index2: int
    target: int


wordsim_tasks: list[WordSimTask] = [
    WordSimTask(
        task="RareWord",
        file="evaluation/wordsim/data/rw.txt",
        index1=0,
        index2=1,
        target=2,
    ),
    WordSimTask(
        task="MEN",
        file="evaluation/wordsim/data/men.txt",
        index1=0,
        index2=1,
        target=2,
    ),
    WordSimTask(
        task="SimLex",
        file="evaluation/wordsim/data/simLex.txt",
        index1=1,
        index2=2,
        target=3,
    ),
    WordSimTask(
        task="rel353",
        file="evaluation/wordsim/data/rel353.txt",
        index1=1,
        index2=2,
        target=3,
    ),
    WordSimTask(
        task="simverb",
        file="evaluation/wordsim/data/simverb_3500.txt",
        index1=2,
        index2=3,
        target=1,
    ),
    WordSimTask(
        task="muturk",
        file="evaluation/wordsim/data/mturk_771.txt",
        index1=1,
        index2=2,
        target=3,
    ),
    WordSimTask(
        task="Card660",
        file="evaluation/wordsim/data/card_660.txt",
        index1=0,
        index2=1,
        target=2,
    ),
]
