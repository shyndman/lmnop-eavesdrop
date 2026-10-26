"""HTTP transport, credential loading, and validated Langfuse operations."""

from __future__ import annotations

import base64
import json
import logging
import os
from http.client import HTTPResponse
from typing import cast
from urllib.request import Request, urlopen

from pydantic import JsonValue, RootModel, TypeAdapter

from langfuse_models.models import (
    LLM_CONNECTIONS_PATH,
    MODELS_PATH,
    OPENROUTER_PROVIDER_NAME,
    USER_AGENT,
    ConnectionPayload,
    CreateModelRequest,
    ExistingModelDefinition,
    LlmConnection,
    LlmConnectionsResponse,
    PaginatedModelsResponse,
)

logger = logging.getLogger(__name__)


def require_env(name: str) -> str:
    """Read a required nonempty environment value.

    :param name: Environment variable name.
    :return: The configured value, without transformation.
    :raises RuntimeError: If the variable is absent or empty.
    """
    value = os.getenv(name)
    if value:
        return value
    logger.error("Missing required environment variable: %s", name)
    raise RuntimeError(f"Missing required environment variable: {name}")


def get_langfuse_base_url() -> str:
    """Resolve the endpoint from LANGFUSE_BASE_URL, then LANGFUSE_HOST.

    :return: URL without trailing slashes, or an empty string if unconfigured.
    """
    return (os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "").rstrip(
        "/"
    )


def build_langfuse_headers() -> dict[str, str]:
    """Build JSON headers authenticated with the Langfuse key pair.

    :return: Basic authorization and JSON content negotiation headers.
    :raises RuntimeError: If LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is missing.
    """
    public_key = require_env("LANGFUSE_PUBLIC_KEY")
    secret_key = require_env("LANGFUSE_SECRET_KEY")
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    return {
        "Accept": "application/json",
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }


def fetch_json[T](
    url: str,
    *,
    headers: dict[str, str],
    response_schema: TypeAdapter[T],
    method: str = "GET",
    payload: CreateModelRequest | ConnectionPayload | None = None,
) -> T:
    """Send JSON and strictly validate the external response before returning it.

    :param url: Complete endpoint URL, including query parameters.
    :param headers: Authentication and content negotiation headers.
    :param response_schema: Schema for the endpoint's JSON response.
    :param method: HTTP verb; callers are responsible for dry-run write guards.
    :param payload: Optional JSON request body.
    :return: Validated response; unknown server fields are discarded.
    :raises pydantic.ValidationError: If known response fields violate the schema.
    :raises urllib.error.URLError: If the request fails (30-second timeout).
    """
    request_data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=request_data, headers=headers, method=method)
    try:
        response = cast(HTTPResponse, urlopen(request, timeout=30))
        with response:
            return response_schema.validate_json(response.read(), strict=True)
    except Exception:
        logger.exception(
            "Model synchronization HTTP request failed: %s %s", method, url
        )
        raise


def fetch_openrouter_connection(
    base_url: str, headers: dict[str, str]
) -> LlmConnection | None:
    """Read the unique safely writable OpenRouter connection.

    :param base_url: Langfuse origin without a trailing slash.
    :param headers: Authenticated request headers.
    :return: Existing connection, or None if absent.
    :raises RuntimeError: If duplicates or hidden extra headers prevent safe writes.
    :raises pydantic.ValidationError: If the response is malformed.
    :raises urllib.error.URLError: If the request fails.
    """
    response = fetch_json(
        f"{base_url}{LLM_CONNECTIONS_PATH}?limit=100&page=1",
        headers=headers,
        response_schema=TypeAdapter(LlmConnectionsResponse),
    )
    matches = [
        item
        for item in response["data"]
        if item["provider"] == OPENROUTER_PROVIDER_NAME
    ]
    if len(matches) > 1:
        logger.error("Multiple OpenRouter connections prevent safe synchronization")
        raise RuntimeError("Expected at most one OpenRouter LLM connection")
    connection = matches[0] if matches else None
    if connection is not None and connection["extraHeaderKeys"]:
        logger.error("OpenRouter connection has hidden extra headers")
        raise RuntimeError(
            "Refusing to overwrite an OpenRouter connection with hidden extra headers"
        )
    return connection


def upsert_openrouter_connection(
    base_url: str,
    headers: dict[str, str],
    payload: ConnectionPayload,
) -> LlmConnection:
    """Persist connection settings and credentials using PUT.

    :param base_url: Langfuse origin without a trailing slash.
    :param headers: Authenticated request headers.
    :param payload: Complete desired connection, including the secret key.
    :return: Validated persisted connection.
    :raises pydantic.ValidationError: If the response is malformed.
    :raises urllib.error.URLError: If the write fails.
    """
    return fetch_json(
        f"{base_url}{LLM_CONNECTIONS_PATH}",
        headers=headers,
        method="PUT",
        payload=payload,
        response_schema=TypeAdapter(LlmConnection),
    )


def fetch_all_model_definitions(
    base_url: str, headers: dict[str, str]
) -> list[ExistingModelDefinition]:
    """Read all model definitions in server pagination order.

    :param base_url: Langfuse origin without a trailing slash.
    :param headers: Authenticated request headers.
    :return: Definitions from every page, including server-managed definitions.
    :raises pydantic.ValidationError: If any page is malformed.
    :raises urllib.error.URLError: If any page request fails.
    """
    models: list[ExistingModelDefinition] = []
    page = 1
    while True:
        response = fetch_json(
            f"{base_url}{MODELS_PATH}?limit=100&page={page}",
            headers=headers,
            response_schema=TypeAdapter(PaginatedModelsResponse),
        )
        models.extend(response["data"])
        if page >= response["meta"]["totalPages"]:
            return models
        page += 1


def create_model_definition(
    base_url: str,
    headers: dict[str, str],
    payload: CreateModelRequest,
) -> ExistingModelDefinition:
    """Create one exact-match priced model definition.

    :param base_url: Langfuse origin without a trailing slash.
    :param headers: Authenticated request headers.
    :param payload: Model definition with per-token pricing.
    :return: Validated persisted definition, including its identifier.
    :raises pydantic.ValidationError: If the response is malformed.
    :raises urllib.error.URLError: If creation fails.
    """
    return fetch_json(
        f"{base_url}{MODELS_PATH}",
        headers=headers,
        method="POST",
        payload=payload,
        response_schema=TypeAdapter(ExistingModelDefinition),
    )


def delete_model_definition(
    base_url: str, headers: dict[str, str], model_id: str
) -> None:
    """Delete a previously ownership-checked definition.

    :param base_url: Langfuse origin without a trailing slash.
    :param headers: Authenticated request headers.
    :param model_id: Server identifier to delete; ownership is the caller's duty.
    :return: None after successful deletion.
    :raises pydantic.ValidationError: If the response is not JSON.
    :raises urllib.error.URLError: If deletion fails.
    """
    _ = fetch_json(
        f"{base_url}{MODELS_PATH}/{model_id}",
        headers=headers,
        method="DELETE",
        response_schema=TypeAdapter(RootModel[JsonValue]),
    )
