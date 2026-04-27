from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class LiteRtRewriteProvider(BaseModel):
  model_config: ClassVar[ConfigDict] = ConfigDict(strict=True, extra="forbid")

  type: Literal["litert"]
  model_path: str = Field(min_length=1)


class PydanticAiRewriteProvider(BaseModel):
  model_config: ClassVar[ConfigDict] = ConfigDict(strict=True, extra="forbid")

  type: Literal["pydantic_ai"]
  model: str = Field(min_length=1)


RewriteProvider = Annotated[
  LiteRtRewriteProvider | PydanticAiRewriteProvider,
  Field(discriminator="type"),
]


class LlmRewriteConfig(BaseModel):
  model_config: ClassVar[ConfigDict] = ConfigDict(strict=True, extra="forbid")

  prompt_path: str = Field(min_length=1)
  provider: RewriteProvider


class ActiveListenerConfig(BaseModel):
  """Validated runtime configuration for the active-listener service.

  :param keyboard_name: Exact evdev device name to capture during dictation.
  :type keyboard_name: str
  :param host: Eavesdrop server hostname.
  :type host: str
  :param port: Eavesdrop server port.
  :type port: int
  :param audio_device: PortAudio capture device name passed to the client.
  :type audio_device: str
  :param ffmpeg_path: Optional explicit FFmpeg binary override.
  :type ffmpeg_path: str | None
  :param llm_rewrite: Nested rewrite configuration.
  :type llm_rewrite: LlmRewriteConfig | None
  """

  model_config: ClassVar[ConfigDict] = ConfigDict(strict=True, extra="forbid")

  keyboard_name: str = Field(min_length=1)
  host: str = Field(min_length=1)
  port: int = Field(ge=1, le=65535)
  audio_device: str = Field(min_length=1)
  ffmpeg_path: str | None = None
  llm_rewrite: LlmRewriteConfig | None = None
