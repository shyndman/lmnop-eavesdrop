"""Guard ownership and semantic equality before applying connection/model changes."""

from __future__ import annotations

import logging
from decimal import Decimal

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
    ConnectionPayload,
    ConnectionState,
    ConnectionSyncResult,
    CreateModelRequest,
    ExistingModelDefinition,
    ExistingPricingTier,
    LlmConnection,
    ModelSignature,
    ModelSyncResult,
    PricingTierInput,
    PricingTierSignature,
)

logger = logging.getLogger(__name__)


def build_connection_payload(
    custom_models: list[str], connection: LlmConnection | None
) -> ConnectionPayload:
    """Build complete desired settings while preserving adapter and config.

    :param custom_models: Catalog model identifiers to expose.
    :param connection: Current connection, or None when creating.
    :return: PUT payload including the required OpenRouter secret.
    :raises RuntimeError: If OPENROUTER_API_KEY is missing.
    """
    payload: ConnectionPayload = {
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


def normalize_connection_state(payload: ConnectionState) -> ConnectionState:
    """Exclude secrets and metadata and canonicalize model ordering.

    :param payload: Desired or existing public connection settings.
    :return: Comparable settings; the input is not mutated.
    """
    normalized: ConnectionState = {
        "provider": payload["provider"],
        "adapter": payload["adapter"],
        "baseURL": payload["baseURL"],
        "customModels": sorted(payload["customModels"]),
        "withDefaultModels": payload["withDefaultModels"],
    }
    if "config" in payload:
        normalized["config"] = payload["config"]

    return normalized


def has_sync_tier(model: ExistingModelDefinition) -> bool:
    """Identify a definition claiming this script's pricing marker.

    :param model: Validated server definition.
    :return: Whether any tier carries the synchronization marker.
    """
    return any(
        tier.get("name") == SYNC_TIER_NAME for tier in model.get("pricingTiers", [])
    )


def is_script_managed_model(model: ExistingModelDefinition) -> bool:
    """Check every ownership condition before a definition can be deleted.

    :param model: Validated server definition.
    :return: Whether this is an exact-match, token-priced custom sync model.
    """
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
    """Index custom definitions, rejecting ambiguous duplicate names.

    :param models: All server definitions, including managed ones.
    :return: Custom definitions keyed by model name.
    :raises RuntimeError: If custom names are duplicated.
    """
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
        logger.error("Duplicate custom model definitions: %s", duplicates)
        raise RuntimeError(
            "Duplicate custom model definitions exist: "
            + ", ".join(sorted(set(duplicates)))
        )

    return indexed


def ensure_safe_custom_models(
    custom_models_by_name: dict[str, ExistingModelDefinition],
    desired_model_names: set[str],
) -> None:
    """Reject conflicting ownership before any model writes occur.

    :param custom_models_by_name: Unique existing custom definitions.
    :param desired_model_names: Catalog names the script intends to own.
    :return: None when every potentially affected definition is safe.
    :raises RuntimeError: If a marker is misused or a desired name is unowned.
    """
    for model in custom_models_by_name.values():
        model_name = model["modelName"]
        if has_sync_tier(model) and not is_script_managed_model(model):
            logger.error("Invalid synchronization ownership for %s", model_name)
            raise RuntimeError(
                f"Custom model definition uses sync marker but is not script-managed: {model_name}"
            )
        if model_name in desired_model_names and not is_script_managed_model(model):
            logger.error("Unowned custom model conflicts with %s", model_name)
            raise RuntimeError(
                f"Custom model definition already exists for {model_name}; refusing to create a duplicate"
            )


def decimal_string(value: float) -> str:
    """Normalize a numeric price without scientific notation or trailing zeroes.

    :param value: Numeric price returned by the catalog or server.
    :return: Canonical decimal string used for semantic comparisons.
    """
    text = format(Decimal(str(value)), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def normalize_price_map(prices: dict[str, float]) -> dict[str, str]:
    """Canonicalize all numeric prices in deterministic key order.

    :param prices: Per-token prices keyed by usage type.
    :return: Canonical decimal strings, preserving all price keys.
    """
    normalized: dict[str, str] = {}
    for key in sorted(prices):
        normalized[key] = decimal_string(prices[key])
    return normalized


def pricing_tier_signature(
    tier: PricingTierInput | ExistingPricingTier,
) -> PricingTierSignature:
    """Project the shared desired/existing tier fields for comparison.

    :param tier: Desired tier or validated persisted tier.
    :return: Comparable fields with normalized prices and no server identifier.
    """
    return {
        "name": tier["name"],
        "isDefault": tier["isDefault"],
        "priority": tier["priority"],
        "conditions": tier["conditions"],
        "prices": normalize_price_map(tier["prices"]),
    }


def desired_model_signature(model: CreateModelRequest) -> ModelSignature:
    """Project a desired model into its semantic comparison fields.

    :param model: Desired definition containing its single default sync tier.
    :return: Comparable name, pattern, unit, and canonical tier.
    :raises IndexError: If the required tier is absent.
    """
    return {
        "modelName": model["modelName"],
        "matchPattern": model["matchPattern"],
        "unit": model["unit"],
        "pricingTiers": [pricing_tier_signature(model["pricingTiers"][0])],
    }


def existing_model_signature(model: ExistingModelDefinition) -> ModelSignature:
    """Project an ownership-checked existing model for semantic comparison.

    :param model: Persisted definition with its single default sync tier.
    :return: Comparable fields excluding server metadata.
    :raises IndexError: If called without first establishing tier ownership.
    """
    return {
        "modelName": model["modelName"],
        "matchPattern": model["matchPattern"],
        "unit": model.get("unit"),
        "pricingTiers": [pricing_tier_signature(model["pricingTiers"][0])],
    }


def sync_connection(
    *,
    base_url: str,
    headers: dict[str, str],
    custom_models: list[str],
    current_connection: LlmConnection | None,
    apply: bool,
) -> ConnectionSyncResult:
    """Compare connection state and optionally persist it.

    :param base_url: Langfuse origin without a trailing slash.
    :param headers: Authenticated request headers.
    :param custom_models: All catalog identifiers, including unpriced models.
    :param current_connection: Existing safe connection, or None.
    :param apply: Allow PUT only when true; dry runs still require credentials.
    :return: Change flag, model count, and available persisted identity.
    :raises RuntimeError: If the OpenRouter secret is missing.
    :raises pydantic.ValidationError: If a write response is malformed.
    :raises urllib.error.URLError: If a write fails.
    """
    desired_payload = build_connection_payload(custom_models, current_connection)
    desired_state = normalize_connection_state(desired_payload)
    current_state = None
    if current_connection is not None:
        current_payload: ConnectionState = {
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
    result: ConnectionSyncResult = {
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


def _partition_model_changes(
    desired_models: dict[str, CreateModelRequest],
    existing_models: dict[str, ExistingModelDefinition],
) -> tuple[list[str], list[str], list[str]]:
    create: list[str] = []
    update: list[str] = []
    unchanged: list[str] = []
    for name in sorted(desired_models):
        existing = existing_models.get(name)
        if existing is None:
            create.append(name)
        elif existing_model_signature(existing) != desired_model_signature(
            desired_models[name]
        ):
            update.append(name)
        else:
            unchanged.append(name)
    return create, update, unchanged


def sync_model_definitions(
    *,
    base_url: str,
    headers: dict[str, str],
    desired_models: dict[str, CreateModelRequest],
    existing_models: list[ExistingModelDefinition],
    apply: bool,
) -> ModelSyncResult:
    """Plan owned model changes and optionally apply delete-then-create writes.

    :param base_url: Langfuse origin without a trailing slash.
    :param headers: Authenticated request headers.
    :param desired_models: Priced catalog definitions keyed by model name.
    :param existing_models: All validated server definitions.
    :param apply: Permit writes only when true; false performs no deletion.
    :return: Sorted create, update, delete plans and desired/existing counts.
    :raises RuntimeError: If custom names duplicate or ownership conflicts.
    :raises pydantic.ValidationError: If a creation response is malformed.
    :raises urllib.error.URLError: If a write fails; earlier writes are not rolled back.
    """
    custom_models_by_name = index_custom_models_by_name(existing_models)
    ensure_safe_custom_models(custom_models_by_name, set(desired_models))

    existing_script_managed_by_name = {
        name: model
        for name, model in custom_models_by_name.items()
        if is_script_managed_model(model)
    }

    create_names, update_names, unchanged_names = _partition_model_changes(
        desired_models, existing_script_managed_by_name
    )

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
