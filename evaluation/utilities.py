from argparse import ArgumentParser
from pathlib import Path
from typing import TypeAlias

from model2vec.model.encoders import StaticEmbedder
from sentence_transformers import SentenceTransformer

Embedder: TypeAlias = StaticEmbedder | SentenceTransformer


def get_default_argparser() -> ArgumentParser:
    """Get the default argument parser."""
    parser = ArgumentParser()
    parser.add_argument("--model-path", help="The model to use.", required=True)
    parser.add_argument("--input", action="store_true")
    parser.add_argument("--word-level", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--suffix", default="")

    return parser


def load_embedder(model_path: str, input_level: bool, word_level: bool, device: str) -> tuple[Embedder, str]:
    """Load the embedder.

    :param model_path: The path to the model.
    :param input_level: Whether to use input level embeddings.
    :param word_level: Whether to use word level embeddings.
    :param device: The device to use.
    :return: The embedder and the name of the model.
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
