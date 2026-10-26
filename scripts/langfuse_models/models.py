"""Typed source and Langfuse payloads, and shared synchronization constants."""

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Required, TypedDict

type JsonValue = (
    None | bool | int | float | str | Sequence[JsonValue] | Mapping[str, JsonValue]
)

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
    input: float
    output: float


class SourceModelEntry(TypedDict, total=False):
    id: Required[str]
    cost: SourceCostEntry


class ProviderEntry(TypedDict):
    id: str
    models: dict[str, SourceModelEntry]


class LlmConnection(TypedDict):
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
    data: list[LlmConnection]


class ExistingPricingTier(TypedDict, total=False):
    id: str
    name: Required[str]
    isDefault: Required[bool]
    priority: Required[int]
    conditions: Required[list[JsonValue]]
    prices: Required[dict[str, float]]


class ExistingModelDefinition(TypedDict, total=False):
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
    page: int
    totalPages: int


class PaginatedModelsResponse(TypedDict):
    data: list[ExistingModelDefinition]
    meta: PaginationMeta


class PricingTierInput(TypedDict):
    name: str
    isDefault: bool
    priority: int
    conditions: list[JsonValue]
    prices: dict[str, float]


class CreateModelRequest(TypedDict):
    modelName: str
    matchPattern: str
    unit: str
    pricingTiers: list[PricingTierInput]
