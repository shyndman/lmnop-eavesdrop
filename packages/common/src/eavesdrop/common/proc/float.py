from collections.abc import MutableMapping
from typing import cast

import numpy as np


class FloatPrecisionProcessor:
  """
  A structlog processor for rounding floats. Both as single numbers or in data structures like
  (nested) lists, dicts, or numpy arrays.

  Inspired by https://github.com/underyx/structlog-pretty/blob/master/structlog_pretty/processors.py

  NOTE: It seems that if a processor logs internally, like for debugging purposes, it will be
  detected and removed from the processing stack.
  """

  def __init__(
    self,
    digits: int = 3,
    only_fields: set[str] | None = None,
    not_fields: set[str] | None = None,
    np_array_to_list: bool = True,
  ) -> None:
    """
    Create a FloatRounder processor. That rounds floats to the given number of digits.

    :param digits: The number of digits to round to
    :param only_fields: A set specifying the fields to round (None = round all fields except
      not_fields)
    :param not_fields: A set specifying fields not to round
    :param np_array_to_list: Whether to cast np.array to list for nicer printing
    """
    self.digits: int = digits
    self.np_array_to_list: bool = np_array_to_list
    self.only_fields: set[str] = only_fields or set()
    self.not_fields: set[str] = not_fields or set()

  def _round(self, value: object) -> object:
    """
    Round floats, unpack lists, convert np.arrays to lists

    :param value: The value/data structure to round
    :returns: The rounded value
    """
    # round floats
    if isinstance(value, float):
      return round(value, self.digits)
    # convert np.array to list
    if self.np_array_to_list:
      if isinstance(value, np.ndarray):
        return self._round(cast(object, value.tolist()))
    # round values in lists recursively (to handle lists of lists)
    if isinstance(value, list):
      list_value = cast(list[object], value)
      for idx, item in enumerate(list_value):
        list_value[idx] = self._round(item)
      return list_value
    # similarly, round values in dicts recursively
    if isinstance(value, dict):
      dict_value = cast(dict[object, object], value)
      for key, item in dict_value.items():
        dict_value[key] = self._round(item)
      return dict_value
    # return any other values as they are
    return value

  def __call__(
    self,
    _: object,
    __: str,
    event_dict: MutableMapping[str, object],
  ) -> MutableMapping[str, object]:
    for key, value in event_dict.items():
      if len(self.only_fields) > 0 and key not in self.only_fields:
        continue
      if len(self.not_fields) > 0 and key in self.not_fields:
        continue
      if isinstance(value, bool):
        continue  # don't convert True to 1.0

      event_dict[key] = self._round(value)
    return event_dict
