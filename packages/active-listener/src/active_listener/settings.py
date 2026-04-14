from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class LlmRewriteConfig(BaseModel):
  model_config: ClassVar[ConfigDict] = ConfigDict(strict=True)

  enabled: bool
  base_url: str = Field(min_length=1)
  timeout_s: int = Field(default=30, ge=1)
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
  :param ydotool_socket: Optional custom ydotool daemon socket path.
  :type ydotool_socket: str | None
  :param llm_rewrite: Nested rewrite configuration.
  :type llm_rewrite: LlmRewriteConfig
  """

  model_config: ClassVar[ConfigDict] = ConfigDict(strict=True)

  keyboard_name: str = Field(min_length=1)
  host: str = Field(min_length=1)
  port: int = Field(ge=1, le=65535)
  audio_device: str = Field(min_length=1)
  ydotool_socket: str | None = None
  llm_rewrite: LlmRewriteConfig
