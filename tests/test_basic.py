"""Basic tests for lmnop:eavesdrop."""

import eavesdrop


def test_import():
  """Test that the module can be imported."""
  assert eavesdrop is not None


def test_version():
  """Test that version is defined."""
  assert hasattr(eavesdrop, "__version__")
