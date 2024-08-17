from pathlib import Path
from typing import TypeAlias

from model2vec.model.encoders import StaticEmbedder
from mteb.abstasks import AbsTask
from mteb.evaluation import MTEB
from sentence_transformers import SentenceTransformer

Embedder: TypeAlias = StaticEmbedder | SentenceTransformer
import logging
from typing import Any

from model2vec.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


class CustomMTEB(MTEB):
    def select_tasks(self, **kwargs: Any) -> None:
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
        super().select_tasks(**kwargs)

    @property
    def available_task_types(self) -> set[str]:
        """Override to ensure task types are gathered from the instances."""
        return {task.metadata_dict["type"] for task in self.tasks}


def load_embedder(model_path: str, input_level: bool, word_level: bool, device: str) -> tuple[Embedder, str]:
    """
    Load the embedder.

    :param model_path: The path to the model.
    :param input_level: Whether to use input level embeddings.
    :param word_level: Whether to use word level embeddings.
    :param device: The device to use.
    :return: The embedder and the name of the model.
    :raises ValueError: If both input and word level are passed.
    """
    embedder: Embedder

    if input_level and word_level:
        raise ValueError("Both input and word level were passed.")

    if input_level:
        embedder = StaticEmbedder.from_model(model_path)
        name = embedder.name
    elif word_level:
        embedder = StaticEmbedder.from_vectors(model_path, apply_pca=True, apply_zipf=True)
        name = embedder.name
    else:
        # Always load on CPU
        embedder = SentenceTransformer(model_name_or_path=model_path, device="cpu")
        embedder = embedder.eval().to(device)
        model_name = Path(model_path).name.replace("_", "-")
        name = f"sentencetransformer_{model_name}"

    return embedder, name
