from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import requests

logger = logging.getLogger(__name__)


class EmbeddingClientProtocol(Protocol):
    @property
    def embedding_dim(self) -> Optional[int]:
        ...

    def embeddings(
        self,
        text: Union[str, List[str], Tuple[str, ...]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        ...


@dataclass
class OpenAICompatibleEmbeddingClient:
    model: str
    api_base: str
    api_key: Optional[str] = None
    embedding_batch_size: int = 32
    embedding_path: str = "/embeddings"
    encoding_format: str = "float"
    request_timeout: float = 120.0
    _embedding_dim: Optional[int] = field(default=None, init=False, repr=False)

    @property
    def endpoint_url(self) -> str:
        return f"{self.api_base.rstrip('/')}/{self.embedding_path.lstrip('/')}"

    @property
    def embedding_dim(self) -> Optional[int]:
        if self._embedding_dim is not None:
            return self._embedding_dim

        if (
            "ollama" in self.api_base
            or (
                ("localhost" in self.api_base or "127.0.0.1" in self.api_base)
                and self.api_key == "ollama"
            )
        ):
            ollama_url_without_v1 = self.api_base.replace("/v1", "")
            response = requests.post(
                f"{ollama_url_without_v1}/api/show",
                json={"name": self.model},
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            model_info = payload.get("model_info", {})

            embedding_key = [key for key in model_info.keys() if "embedding_length" in key]
            if len(embedding_key) == 1:
                self._embedding_dim = int(model_info[embedding_key[0]])
                return self._embedding_dim

            raise ValueError(f"Model information not found in Ollama response: {payload}")

        raise NotImplementedError("Embedding dimension retrieval not implemented for this API base")

    def embeddings(
        self,
        text: Union[str, List[str], Tuple[str, ...]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if batch_size is None:
            batch_size = self.embedding_batch_size

        if isinstance(text, str):
            text = (text,)
        elif isinstance(text, list):
            text = tuple(text)

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        batch_buffer: list[list[float]] = []

        for batch_chunk_idx in range(0, len(text), batch_size):
            batch_chunk = list(text[batch_chunk_idx:batch_chunk_idx + batch_size])
            payload = {
                "model": self.model,
                "input": batch_chunk,
                "encoding_format": self.encoding_format,
            }
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.request_timeout,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                detail = response.text.strip()
                raise RuntimeError(
                    f"Embedding request failed with status {response.status_code} at "
                    f"{self.endpoint_url}: {detail}"
                ) from exc

            batch_buffer.extend(self._parse_embeddings_response(response.json()))

        return np.array(batch_buffer, dtype=np.float32)

    @staticmethod
    def _parse_embeddings_response(payload: object) -> list[list[float]]:
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                return [item["embedding"] for item in data]

            embeddings = payload.get("embeddings")
            if isinstance(embeddings, list):
                if embeddings and isinstance(embeddings[0], dict):
                    return [item["embedding"] for item in embeddings]
                return embeddings

            embedding = payload.get("embedding")
            if isinstance(embedding, list):
                return [embedding]

        raise ValueError(f"Unexpected embeddings response payload: {payload}")
