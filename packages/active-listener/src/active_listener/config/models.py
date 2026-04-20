from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class LlmRewriteConfig(BaseModel):
  model_config: ClassVar[ConfigDict] = ConfigDict(strict=True, extra="forbid")

  enabled: bool
  model_path: str = Field(min_length=1)
  prompt_path: str = Field(min_length=1)


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
  :param llm_rewrite: Nested rewrite configuration.
  :type llm_rewrite: LlmRewriteConfig
  """

  model_config: ClassVar[ConfigDict] = ConfigDict(strict=True, extra="forbid")

  keyboard_name: str = Field(min_length=1)
  host: str = Field(min_length=1)
  port: int = Field(ge=1, le=65535)
  audio_device: str = Field(min_length=1)
  llm_rewrite: LlmRewriteConfig
