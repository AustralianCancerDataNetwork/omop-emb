from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import requests

from omop_emb.embedding_client import OpenAICompatibleEmbeddingClient


class FakeResponse:
    def __init__(self, *, status_code: int, payload: object, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code} {self.text}")


@pytest.mark.unit
def test_embeddings_single_text_sends_model_and_scalar_input(monkeypatch):
    captured_payloads: list[dict[str, Any]] = []

    def fake_post(url, json, headers, timeout):
        captured_payloads.append(json)
        return FakeResponse(
            status_code=200,
            payload={"data": [{"embedding": [0.1, 0.2, 0.3]}]},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OpenAICompatibleEmbeddingClient(
        model="openai:text-embedding-3-small",
        api_base="http://localhost:8000/v1",
    )

    embeddings = client.embeddings("Hello, world!")

    assert embeddings.shape == (1, 3)
    assert np.allclose(embeddings, np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
    assert captured_payloads == [
        {
            "model": "openai:text-embedding-3-small",
            "input": "Hello, world!",
            "encoding_format": "float",
        }
    ]


@pytest.mark.unit
def test_embeddings_retries_without_encoding_format_when_rejected(monkeypatch):
    captured_payloads: list[dict[str, Any]] = []

    def fake_post(url, json, headers, timeout):
        captured_payloads.append(json)
        if len(captured_payloads) == 1:
            return FakeResponse(
                status_code=400,
                payload={"error": "unsupported field"},
                text="unsupported field: encoding_format",
            )
        return FakeResponse(
            status_code=200,
            payload={"data": [{"embedding": [0.4, 0.5]}]},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    client = OpenAICompatibleEmbeddingClient(
        model="openai:text-embedding-3-small",
        api_base="http://localhost:8000/v1",
    )

    embeddings = client.embeddings("Hello, world!")

    assert embeddings.shape == (1, 2)
    assert captured_payloads == [
        {
            "model": "openai:text-embedding-3-small",
            "input": "Hello, world!",
            "encoding_format": "float",
        },
        {
            "model": "openai:text-embedding-3-small",
            "input": "Hello, world!",
        },
    ]
