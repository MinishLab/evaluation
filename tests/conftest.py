from typing import Any, Sequence
from unittest.mock import create_autospec

import pytest
import torch
from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta


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

    # Set the model meta
    mock_encoder.mteb_model_meta = ModelMeta(
        name="mock_model_name", revision="mock_revision", release_date=None, languages=None
    )

    return mock_encoder
