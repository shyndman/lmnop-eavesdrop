from collections.abc import Sequence
from types import TracebackType
from typing import Literal, NotRequired, Protocol, TypedDict

class TextContent(TypedDict):
  type: Literal["text"]
  text: str

class Message(TypedDict):
  role: Literal["system"]
  content: list[TextContent]

class ResponseContent(TypedDict):
  type: NotRequired[str]
  text: NotRequired[str]

class Response(TypedDict):
  content: NotRequired[list[ResponseContent]]

class Conversation(Protocol):
  def __enter__(self) -> Conversation: ...
  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    traceback: TracebackType | None,
  ) -> bool | None: ...
  def send_message(self, prompt: str) -> Response: ...

class Engine:
  def __init__(self, model_path: str, *, backend: object) -> None: ...
  def __enter__(self) -> Engine: ...
  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    traceback: TracebackType | None,
  ) -> bool | None: ...
  def create_conversation(
    self,
    *,
    messages: Sequence[Message] | None = None,
  ) -> Conversation: ...

class Backend:
  CPU: object
