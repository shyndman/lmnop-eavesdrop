"""Fetch the provider catalog and translate its prices into exact model requests."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import cast

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


def fetch_provider_map() -> dict[str, ProviderEntry]:
    payload = fetch_json(
        MODELS_API_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    if not isinstance(payload, dict):
        raise TypeError("models.dev API returned a non-object payload")

    return cast(dict[str, ProviderEntry], payload)


def build_custom_models(provider_map: dict[str, ProviderEntry]) -> list[str]:
    if OPENROUTER_PROVIDER_KEY not in provider_map:
        raise KeyError(f"Provider not found: {OPENROUTER_PROVIDER_KEY}")

    openrouter_provider = provider_map[OPENROUTER_PROVIDER_KEY]
    return sorted({model["id"] for model in openrouter_provider["models"].values()})


def exact_match_pattern(model_name: str) -> str:
    return f"^{re.escape(model_name)}$"


def normalize_price_per_token(price_per_million: float) -> float:
    return float(Decimal(str(price_per_million)) / ONE_MILLION)


def build_desired_model_definitions(
    provider_map: dict[str, ProviderEntry],
) -> dict[str, CreateModelRequest]:
    if OPENROUTER_PROVIDER_KEY not in provider_map:
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
