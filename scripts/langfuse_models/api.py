"""HTTP transport, credential loading, and typed Langfuse endpoint operations."""

from __future__ import annotations

import base64
import json
import os
from http.client import HTTPResponse
from typing import cast
from urllib.request import Request, urlopen

from langfuse_models.models import (
    LLM_CONNECTIONS_PATH,
    MODELS_PATH,
    OPENROUTER_PROVIDER_NAME,
    USER_AGENT,
    CreateModelRequest,
    ExistingModelDefinition,
    JsonValue,
    LlmConnection,
    LlmConnectionsResponse,
    PaginatedModelsResponse,
)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value

    raise RuntimeError(f"Missing required environment variable: {name}")


def get_langfuse_base_url() -> str:
    return (os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "").rstrip(
        "/"
    )


def build_langfuse_headers() -> dict[str, str]:
    public_key = require_env("LANGFUSE_PUBLIC_KEY")
    secret_key = require_env("LANGFUSE_SECRET_KEY")
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    return {
        "Accept": "application/json",
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }


def fetch_json(
    url: str,
    *,
    headers: dict[str, str],
    method: str = "GET",
    payload: CreateModelRequest | dict[str, JsonValue] | None = None,
) -> object:
    """Decode raw HTTP JSON; endpoint callers apply their response contracts.

    :param method: HTTP verb; callers enforce synchronization write guards.
    :param payload: Optional JSON request body.
    """
    request_data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=request_data, headers=headers, method=method)
    with cast(HTTPResponse, urlopen(request, timeout=30)) as response:
        return cast(object, json.load(response))


def fetch_openrouter_connection(
    base_url: str, headers: dict[str, str]
) -> LlmConnection | None:
    payload = fetch_json(
        f"{base_url}{LLM_CONNECTIONS_PATH}?limit=100&page=1",
        headers=headers,
    )
    if not isinstance(payload, dict):
        raise TypeError("Langfuse returned a non-object llm-connections payload")

    response = cast(LlmConnectionsResponse, payload)
    matches = [
        connection
        for connection in response["data"]
        if connection["provider"] == OPENROUTER_PROVIDER_NAME
    ]
    if len(matches) > 1:
        raise RuntimeError("Expected at most one OpenRouter LLM connection")

    connection = matches[0] if matches else None
    if connection is not None and connection["extraHeaderKeys"]:
        raise RuntimeError(
            "Refusing to overwrite an OpenRouter connection with hidden extra headers"
        )

    return connection


def upsert_openrouter_connection(
    base_url: str, headers: dict[str, str], payload: dict[str, JsonValue]
) -> LlmConnection:
    response = fetch_json(
        f"{base_url}{LLM_CONNECTIONS_PATH}",
        headers=headers,
        method="PUT",
        payload=payload,
    )
    if not isinstance(response, dict):
        raise TypeError("Langfuse returned a non-object llm connection upsert response")

    return cast(LlmConnection, response)


def fetch_all_model_definitions(
    base_url: str, headers: dict[str, str]
) -> list[ExistingModelDefinition]:
    models: list[ExistingModelDefinition] = []
    page = 1
    while True:
        payload = fetch_json(
            f"{base_url}{MODELS_PATH}?limit=100&page={page}",
            headers=headers,
        )
        if not isinstance(payload, dict):
            raise TypeError("Langfuse returned a non-object models payload")

        response = cast(PaginatedModelsResponse, payload)
        models.extend(response["data"])
        if page >= response["meta"]["totalPages"]:
            return models
        page += 1


def create_model_definition(
    base_url: str, headers: dict[str, str], payload: CreateModelRequest
) -> ExistingModelDefinition:
    response = fetch_json(
        f"{base_url}{MODELS_PATH}",
        headers=headers,
        method="POST",
        payload=payload,
    )
    if not isinstance(response, dict):
        raise TypeError("Langfuse returned a non-object model create response")

    return cast(ExistingModelDefinition, response)


def delete_model_definition(
    base_url: str, headers: dict[str, str], model_id: str
) -> None:
    _ = fetch_json(
        f"{base_url}{MODELS_PATH}/{model_id}",
        headers=headers,
        method="DELETE",
    )
