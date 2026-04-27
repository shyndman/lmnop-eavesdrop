from typing import NamedTuple, override

from rich.pretty import pretty_repr


class Pretty(NamedTuple):
  value: object

  @override
  def __str__(self) -> str:
    return pretty_repr(self.value)


class Unit(NamedTuple):
  value: float


class Seconds(Unit):
  @override
  def __str__(self) -> str:
    return f"{self.value:.3}s"


class Milliseconds(Unit):
  @override
  def __str__(self) -> str:
    return f"{self.value:.3}ms"


class Microseconds(Unit):
  @override
  def __str__(self) -> str:
    return f"{self.value:.3}μs"


class Samples(Unit):
  @override
  def __str__(self) -> str:
    return f"{self.value:.3} samples"


_EN_DASH = "–"


class Range(NamedTuple):
  start: Unit
  end: Unit

  @override
  def __str__(self) -> str:
    return f"{self.start}{_EN_DASH}{self.end}"
