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
        original_was_scalar = isinstance(text, str)
        if batch_size is None:
            batch_size = self.embedding_batch_size

        if original_was_scalar:
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
            response = self._post_embeddings_request(
                batch_chunk=batch_chunk,
                single_input=len(batch_chunk) == 1 and original_was_scalar,
                headers=headers,
            )

            batch_buffer.extend(self._parse_embeddings_response(response.json()))

        return np.array(batch_buffer, dtype=np.float32)

    def _post_embeddings_request(
        self,
        *,
        batch_chunk: list[str],
        single_input: bool,
        headers: dict[str, str],
    ) -> requests.Response:
        primary_payload = self._build_embeddings_payload(
            batch_chunk=batch_chunk,
            single_input=single_input,
            include_encoding_format=True,
        )
        response = requests.post(
            self.endpoint_url,
            json=primary_payload,
            headers=headers,
            timeout=self.request_timeout,
        )
        if response.ok:
            return response

        if self.encoding_format:
            fallback_payload = self._build_embeddings_payload(
                batch_chunk=batch_chunk,
                single_input=single_input,
                include_encoding_format=False,
            )
            fallback_response = requests.post(
                self.endpoint_url,
                json=fallback_payload,
                headers=headers,
                timeout=self.request_timeout,
            )
            if fallback_response.ok:
                return fallback_response

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            raise RuntimeError(
                f"Embedding request failed with status {response.status_code} at "
                f"{self.endpoint_url}: {detail}"
            ) from exc
        return response

    def _build_embeddings_payload(
        self,
        *,
        batch_chunk: list[str],
        single_input: bool,
        include_encoding_format: bool,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "input": batch_chunk[0] if single_input else batch_chunk,
        }
        if include_encoding_format and self.encoding_format:
            payload["encoding_format"] = self.encoding_format
        return payload

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
