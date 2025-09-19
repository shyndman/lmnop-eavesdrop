from typing import NamedTuple

from rich.pretty import pretty_repr


class Pretty(NamedTuple):
  value: object

  def __str__(self) -> str:
    return pretty_repr(self.value)


class Unit(NamedTuple):
  value: float


class Seconds(Unit):
  def __str__(self) -> str:
    return f"{self.value:.3}s"


class Milliseconds(Unit):
  def __str__(self) -> str:
    return f"{self.value:.3}ms"


class Microseconds(Unit):
  def __str__(self) -> str:
    return f"{self.value:.3}μs"


class Samples(Unit):
  def __str__(self) -> str:
    return f"{self.value:.3} samples"


_EN_DASH = "–"


class Range(NamedTuple):
  start: Unit
  end: Unit

  def __str__(self) -> str:
    return f"{self.start}{_EN_DASH}{self.end}"
