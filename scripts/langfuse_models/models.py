"""Typed source and Langfuse payloads, and shared synchronization constants."""

from decimal import Decimal
from typing import NotRequired, Required, TypedDict

from pydantic import JsonValue

MODELS_API_URL = "https://models.dev/api.json"
LLM_CONNECTIONS_PATH = "/api/public/llm-connections"
MODELS_PATH = "/api/public/models"
OPENROUTER_PROVIDER_KEY = "openrouter"
OPENROUTER_PROVIDER_NAME = "OpenRouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SYNC_TIER_NAME = "models.dev sync"
TOKENS_UNIT = "TOKENS"
USER_AGENT = "eavesdrop-langfuse-model-sync/0.2"
ONE_MILLION = Decimal(1000000)


class SourceCostEntry(TypedDict, total=False):
    """Catalog prices in currency units per million tokens.

    :ivar input: Optional input-token price.
    :ivar output: Optional output-token price.
    """

    input: float
    output: float


class SourceModelEntry(TypedDict, total=False):
    """Known model catalog fields validated before price projection.

    :ivar id: Provider's model identifier.
    :ivar cost: Optional pricing; absent/null means no priced definition.
    """

    id: Required[str]
    cost: SourceCostEntry | None


class ProviderEntry(TypedDict):
    """Provider catalog entry.

    :ivar id: Provider identifier.
    :ivar models: Models keyed by their catalog identifiers.
    """

    id: str
    models: dict[str, SourceModelEntry]


class LlmConnection(TypedDict):
    """Validated public Langfuse connection, never containing its stored secret.

    :ivar extraHeaderKeys: Hidden header names; nonempty values forbid overwrite.
    :ivar config: Adapter settings preserved when synchronizing.
    :ivar updatedAt: Persisted modification timestamp reported in sync results.
    """

    id: str
    provider: str
    adapter: str
    baseURL: str | None
    customModels: list[str]
    withDefaultModels: bool
    extraHeaderKeys: list[str]
    config: dict[str, JsonValue] | None
    updatedAt: str


class LlmConnectionsResponse(TypedDict):
    """Connection-list response.

    :ivar data: Public connection settings returned by Langfuse.
    """

    data: list[LlmConnection]


class ExistingPricingTier(TypedDict, total=False):
    """Persisted tier with numeric per-token prices and optional server identity.

    :ivar conditions: Tier applicability constraints, retained for ownership checks.
    :ivar prices: Usage-type prices normalized before equality comparison.
    """

    id: str
    name: Required[str]
    isDefault: Required[bool]
    priority: Required[int]
    conditions: Required[list[JsonValue]]
    prices: Required[dict[str, float]]


class ExistingModelDefinition(TypedDict, total=False):
    """Validated persisted definition, including fields used to establish ownership.

    :ivar isLangfuseManaged: Server ownership marker; omission is treated as managed.
    :ivar pricingTiers: Persisted tiers checked before any delete or replacement.
    """

    id: Required[str]
    modelName: Required[str]
    matchPattern: Required[str]
    unit: str | None
    tokenizerId: str | None
    tokenizerConfig: JsonValue | None
    startDate: str | None
    isLangfuseManaged: bool
    pricingTiers: Required[list[ExistingPricingTier]]
    createdAt: str


class PaginationMeta(TypedDict):
    """Pagination state returned by the models endpoint.

    :ivar page: Current response page.
    :ivar totalPages: Last page that must be fetched.
    """

    page: int
    totalPages: int


class PaginatedModelsResponse(TypedDict):
    """One validated model-definition page.

    :ivar data: Definitions in server response order.
    :ivar meta: Pagination state for fetching the remaining pages.
    """

    data: list[ExistingModelDefinition]
    meta: PaginationMeta


class PricingTierInput(TypedDict):
    """Tier creation fields, without persisted server metadata.

    :ivar prices: Numeric prices per single token, not per million.
    """

    name: str
    isDefault: bool
    priority: int
    conditions: list[JsonValue]
    prices: dict[str, float]


class CreateModelRequest(TypedDict):
    """Exact-match model creation payload.

    :ivar matchPattern: Anchored literal model identifier.
    :ivar pricingTiers: Single default tier carrying the script ownership marker.
    """

    modelName: str
    matchPattern: str
    unit: str
    pricingTiers: list[PricingTierInput]


class ConnectionState(TypedDict):
    """Comparable public connection settings, excluding credentials and metadata."""

    provider: str
    adapter: str
    baseURL: str | None
    customModels: list[str]
    withDefaultModels: bool
    config: NotRequired[dict[str, JsonValue]]


class ConnectionPayload(ConnectionState):
    """Connection settings plus the required OpenRouter credential."""

    secretKey: str


class ConnectionSyncResult(TypedDict):
    """Connection plan and, when available, persisted identity."""

    willUpdate: bool
    customModelsCount: int
    id: NotRequired[str]
    updatedAt: NotRequired[str]


class ModelSyncResult(TypedDict):
    """Deterministically ordered model changes and summary counts."""

    desiredCount: int
    existingManagedCount: int
    create: list[str]
    update: list[str]
    delete: list[str]
    unchangedCount: int


class PricingTierSignature(TypedDict):
    """Comparable tier settings with exact decimal price strings."""

    name: str
    isDefault: bool
    priority: int
    conditions: list[JsonValue]
    prices: dict[str, str]


class ModelSignature(TypedDict):
    """Comparable model fields, excluding server identity and timestamps."""

    modelName: str
    matchPattern: str
    unit: str | None
    pricingTiers: list[PricingTierSignature]
