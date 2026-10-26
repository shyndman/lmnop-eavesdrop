"""Fetch the provider catalog and translate its prices into exact model requests."""

from __future__ import annotations

import logging
import re
from decimal import Decimal

from pydantic import TypeAdapter

from langfuse_models.api import fetch_json
from langfuse_models.models import (
    MODELS_API_URL,
    ONE_MILLION,
    OPENROUTER_PROVIDER_KEY,
    SYNC_TIER_NAME,
    TOKENS_UNIT,
    USER_AGENT,
    CreateModelRequest,
    ProviderEntry,
)

logger = logging.getLogger(__name__)


def fetch_provider_map() -> dict[str, ProviderEntry]:
    """Load and strictly validate the models.dev provider catalog.

    :return: Provider identifiers mapped to validated catalog entries.
    :raises pydantic.ValidationError: If known catalog fields are malformed.
    :raises urllib.error.URLError: If the catalog request fails.
    """
    payload = fetch_json(
        MODELS_API_URL,
        response_schema=TypeAdapter(dict[str, ProviderEntry]),
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    return payload


def build_custom_models(provider_map: dict[str, ProviderEntry]) -> list[str]:
    """Collect every OpenRouter model, including models without pricing.

    :param provider_map: Validated provider catalog.
    :return: Sorted unique model identifiers.
    :raises KeyError: If OpenRouter is absent.
    """
    if OPENROUTER_PROVIDER_KEY not in provider_map:
        logger.error("Catalog is missing provider %s", OPENROUTER_PROVIDER_KEY)
        raise KeyError(f"Provider not found: {OPENROUTER_PROVIDER_KEY}")

    openrouter_provider = provider_map[OPENROUTER_PROVIDER_KEY]
    return sorted({model["id"] for model in openrouter_provider["models"].values()})


def exact_match_pattern(model_name: str) -> str:
    """Build an anchored literal model-name regular expression.

    :param model_name: Identifier whose metacharacters must be escaped.
    :return: Pattern matching only the complete identifier.
    """
    return f"^{re.escape(model_name)}$"


def normalize_price_per_token(price_per_million: float) -> float:
    """Convert catalog pricing from million-token to single-token units.

    :param price_per_million: Catalog price for one million tokens.
    :return: Per-token price computed through decimal arithmetic.
    """
    return float(Decimal(str(price_per_million)) / ONE_MILLION)


def build_desired_model_definitions(
    provider_map: dict[str, ProviderEntry],
) -> dict[str, CreateModelRequest]:
    """Project priced OpenRouter models into owned Langfuse definitions.

    :param provider_map: Validated provider catalog.
    :return: Exact-match definitions with one default sync tier; unpriced models
        are omitted and missing individual prices remain absent.
    :raises KeyError: If OpenRouter is absent.
    """
    if OPENROUTER_PROVIDER_KEY not in provider_map:
        logger.error("Catalog is missing provider %s", OPENROUTER_PROVIDER_KEY)
        raise KeyError(f"Provider not found: {OPENROUTER_PROVIDER_KEY}")

    desired: dict[str, CreateModelRequest] = {}
    for model in sorted(
        provider_map[OPENROUTER_PROVIDER_KEY]["models"].values(),
        key=lambda item: item["id"],
    ):
        cost = model.get("cost")
        if cost is None:
            continue

        prices: dict[str, float] = {}
        if "input" in cost:
            prices["input"] = normalize_price_per_token(cost["input"])
        if "output" in cost:
            prices["output"] = normalize_price_per_token(cost["output"])
        if not prices:
            continue

        model_name = model["id"]
        desired[model_name] = {
            "modelName": model_name,
            "matchPattern": exact_match_pattern(model_name),
            "unit": TOKENS_UNIT,
            "pricingTiers": [
                {
                    "name": SYNC_TIER_NAME,
                    "isDefault": True,
                    "priority": 0,
                    "conditions": [],
                    "prices": prices,
                }
            ],
        }

    return desired
