"""Guard ownership and semantic equality before applying connection/model changes."""

from __future__ import annotations

from decimal import Decimal
from typing import cast

from langfuse_models.api import (
    create_model_definition,
    delete_model_definition,
    require_env,
    upsert_openrouter_connection,
)
from langfuse_models.catalog import exact_match_pattern
from langfuse_models.models import (
    OPENROUTER_BASE_URL,
    OPENROUTER_PROVIDER_NAME,
    SYNC_TIER_NAME,
    TOKENS_UNIT,
    CreateModelRequest,
    ExistingModelDefinition,
    JsonValue,
    LlmConnection,
)


def build_connection_payload(
    custom_models: list[str], connection: LlmConnection | None
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "provider": OPENROUTER_PROVIDER_NAME,
        "adapter": connection["adapter"] if connection is not None else "openai",
        "secretKey": require_env("OPENROUTER_API_KEY"),
        "baseURL": (
            connection["baseURL"]
            if connection is not None and connection["baseURL"] is not None
            else OPENROUTER_BASE_URL
        ),
        "customModels": custom_models,
        "withDefaultModels": False,
    }
    if connection is not None and connection["config"] is not None:
        payload["config"] = connection["config"]

    return payload


def normalize_connection_state(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    normalized: dict[str, JsonValue] = {
        "provider": payload["provider"],
        "adapter": payload["adapter"],
        "baseURL": payload["baseURL"],
        "customModels": sorted(cast(list[str], payload["customModels"])),
        "withDefaultModels": payload["withDefaultModels"],
    }
    if "config" in payload:
        normalized["config"] = payload["config"]

    return normalized


def has_sync_tier(model: ExistingModelDefinition) -> bool:
    return any(
        tier.get("name") == SYNC_TIER_NAME for tier in model.get("pricingTiers", [])
    )


def is_script_managed_model(model: ExistingModelDefinition) -> bool:
    if model.get("isLangfuseManaged", True):
        return False
    model_name = model.get("modelName")
    if model.get("matchPattern") != exact_match_pattern(model_name):
        return False
    if model.get("unit") != TOKENS_UNIT:
        return False

    pricing_tiers = model.get("pricingTiers", [])
    if len(pricing_tiers) != 1:
        return False

    tier = pricing_tiers[0]
    return (
        tier.get("name") == SYNC_TIER_NAME
        and tier.get("isDefault") is True
        and tier.get("priority") == 0
        and tier.get("conditions") == []
    )


def index_custom_models_by_name(
    models: list[ExistingModelDefinition],
) -> dict[str, ExistingModelDefinition]:
    custom_models = [
        model for model in models if not model.get("isLangfuseManaged", True)
    ]
    indexed: dict[str, ExistingModelDefinition] = {}
    duplicates: list[str] = []
    for model in sorted(custom_models, key=lambda item: item["modelName"]):
        model_name = model["modelName"]
        if model_name in indexed:
            duplicates.append(model_name)
            continue
        indexed[model_name] = model

    if duplicates:
        raise RuntimeError(
            "Duplicate custom model definitions exist: "
            + ", ".join(sorted(set(duplicates)))
        )

    return indexed


def ensure_safe_custom_models(
    custom_models_by_name: dict[str, ExistingModelDefinition],
    desired_model_names: set[str],
) -> None:
    for model in custom_models_by_name.values():
        model_name = model["modelName"]
        if has_sync_tier(model) and not is_script_managed_model(model):
            raise RuntimeError(
                f"Custom model definition uses sync marker but is not script-managed: {model_name}"
            )
        if model_name in desired_model_names and not is_script_managed_model(model):
            raise RuntimeError(
                f"Custom model definition already exists for {model_name}; refusing to create a duplicate"
            )


def decimal_string(value: float) -> str:
    text = format(Decimal(str(value)), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def normalize_price_map(prices: dict[str, float]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key in sorted(prices):
        normalized[key] = decimal_string(prices[key])
    return normalized


def desired_model_signature(model: CreateModelRequest) -> dict[str, JsonValue]:
    tier = model["pricingTiers"][0]
    return {
        "modelName": model["modelName"],
        "matchPattern": model["matchPattern"],
        "unit": model["unit"],
        "pricingTiers": [
            {
                "name": tier["name"],
                "isDefault": tier["isDefault"],
                "priority": tier["priority"],
                "conditions": tier["conditions"],
                "prices": normalize_price_map(tier["prices"]),
            }
        ],
    }


def existing_model_signature(model: ExistingModelDefinition) -> dict[str, JsonValue]:
    tier = model["pricingTiers"][0]
    return {
        "modelName": model["modelName"],
        "matchPattern": model["matchPattern"],
        "unit": model.get("unit"),
        "pricingTiers": [
            {
                "name": tier["name"],
                "isDefault": tier["isDefault"],
                "priority": tier["priority"],
                "conditions": tier["conditions"],
                "prices": normalize_price_map(tier["prices"]),
            }
        ],
    }


def sync_connection(
    *,
    base_url: str,
    headers: dict[str, str],
    custom_models: list[str],
    current_connection: LlmConnection | None,
    apply: bool,
) -> dict[str, JsonValue]:
    """Compare connection state and optionally persist it.

    :param apply: Permit writes only when true; credentials are still required.
    :raises RuntimeError: If the OpenRouter secret is missing.
    """
    desired_payload = build_connection_payload(custom_models, current_connection)
    desired_state = normalize_connection_state(desired_payload)
    current_state = None
    if current_connection is not None:
        current_payload: dict[str, JsonValue] = {
            "provider": current_connection["provider"],
            "adapter": current_connection["adapter"],
            "baseURL": current_connection["baseURL"],
            "customModels": current_connection["customModels"],
            "withDefaultModels": current_connection["withDefaultModels"],
        }
        if current_connection["config"] is not None:
            current_payload["config"] = current_connection["config"]
        current_state = normalize_connection_state(current_payload)

    will_update = current_state != desired_state
    result: dict[str, JsonValue] = {
        "willUpdate": will_update,
        "customModelsCount": len(custom_models),
    }
    if current_connection is not None:
        result["id"] = current_connection["id"]
        result["updatedAt"] = current_connection["updatedAt"]

    if will_update and apply:
        updated_connection = upsert_openrouter_connection(
            base_url, headers, desired_payload
        )
        result["id"] = updated_connection["id"]
        result["updatedAt"] = updated_connection["updatedAt"]

    return result


def sync_model_definitions(
    *,
    base_url: str,
    headers: dict[str, str],
    desired_models: dict[str, CreateModelRequest],
    existing_models: list[ExistingModelDefinition],
    apply: bool,
) -> dict[str, JsonValue]:
    """Plan model changes, rejecting ambiguous ownership before any writes.

    :param apply: Delete obsolete/replaced definitions, then create, only if true.
    :raises RuntimeError: If custom definitions duplicate or conflict with ownership.
    """
    custom_models_by_name = index_custom_models_by_name(existing_models)
    ensure_safe_custom_models(custom_models_by_name, set(desired_models))

    existing_script_managed_by_name = {
        name: model
        for name, model in custom_models_by_name.items()
        if is_script_managed_model(model)
    }

    create_names: list[str] = []
    update_names: list[str] = []
    unchanged_names: list[str] = []
    for model_name in sorted(desired_models):
        existing = existing_script_managed_by_name.get(model_name)
        if existing is None:
            create_names.append(model_name)
            continue
        if existing_model_signature(existing) != desired_model_signature(
            desired_models[model_name]
        ):
            update_names.append(model_name)
            continue
        unchanged_names.append(model_name)

    delete_names = sorted(
        name for name in existing_script_managed_by_name if name not in desired_models
    )

    if apply:
        for model_name in delete_names + update_names:
            delete_model_definition(
                base_url, headers, existing_script_managed_by_name[model_name]["id"]
            )
        for model_name in create_names + update_names:
            _ = create_model_definition(base_url, headers, desired_models[model_name])

    return {
        "desiredCount": len(desired_models),
        "existingManagedCount": len(existing_script_managed_by_name),
        "create": create_names,
        "update": update_names,
        "delete": delete_names,
        "unchangedCount": len(unchanged_names),
    }
