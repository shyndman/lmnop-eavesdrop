"""Contract tests for CLI interface.

Tests the command-line interface contracts including argument parsing,
validation, and command execution patterns using Clypi framework.

CRITICAL: These tests must fail until implementation is complete.
"""

import pytest

from eavesdrop.active_listener.__main__ import ActiveListener


class TestCLIInterface:
  """Test CLI command interface contracts."""

  def test_command_argument_defaults(self):
    """Test that command arguments have expected default values."""
    from eavesdrop.active_listener.__main__ import ServerHostPort

    cmd = ActiveListener.parse([])

    # Default values from specification
    assert cmd.server == ServerHostPort(host="localhost", port=9090)
    assert cmd.audio_device == "default"

  def test_server_parser_valid_formats(self):
    """Test that server parser accepts valid hostname:port formats."""
    from eavesdrop.active_listener.__main__ import ServerHostPort, parse_server

    # Valid formats should parse successfully
    result = parse_server("localhost:9090")
    assert result == ServerHostPort(host="localhost", port=9090)

    result = parse_server("192.168.1.100:8080")
    assert result == ServerHostPort(host="192.168.1.100", port=8080)

    result = parse_server("example.com:443")
    assert result == ServerHostPort(host="example.com", port=443)

  def test_server_parser_invalid_formats(self):
    """Test that server parser rejects invalid formats."""
    from eavesdrop.active_listener.__main__ import parse_server

    # Invalid formats should raise ValueError
    with pytest.raises(ValueError, match="Invalid server format"):
      parse_server("localhost")  # Missing port

    with pytest.raises(ValueError, match="Invalid server format"):
      parse_server(":9090")  # Missing hostname

    with pytest.raises(ValueError, match="Invalid server format"):
      parse_server("localhost:")  # Missing port after colon

    with pytest.raises(ValueError, match="Invalid port"):
      parse_server("localhost:invalid")  # Non-numeric port

    with pytest.raises(ValueError, match="Invalid port"):
      parse_server("localhost:-1")  # Negative port

    with pytest.raises(ValueError, match="Invalid port"):
      parse_server("localhost:70000")  # Port too high

  def test_command_validation_empty_server(self):
    """Test that empty server value is rejected."""
    # This should fail until validation is implemented
    with pytest.raises(SystemExit):
      ActiveListener.parse(["--server", "", "--audio-device", "default"])

  def test_command_help_text(self):
    """Test that command provides help text for arguments."""
    cmd = ActiveListener.parse([])

    # Should have docstring or help attributes
    # This will fail until implementation
    assert cmd.__doc__ is not None
    assert len(cmd.__doc__) > 0
    assert "transcription" in cmd.__doc__.lower()
    assert "eavesdrop" in cmd.__doc__.lower()

  def test_server_parsing_integration(self):
    """Test that server argument uses custom parser correctly."""
    from eavesdrop.active_listener.__main__ import ServerHostPort

    # This should work with valid server format
    cmd = ActiveListener.parse(["--server", "192.168.1.100:8080", "--audio-device", "hw:1,0"])
    assert cmd.server == ServerHostPort(host="192.168.1.100", port=8080)

    # This should fail with invalid format (Clypi exits on parser error)
    with pytest.raises(SystemExit):
      ActiveListener.parse(["--server", "invalid-format", "--audio-device", "default"])

  def test_parsed_server_components_access(self):
    """Test that command can extract host and port from parsed server."""
    from eavesdrop.active_listener.__main__ import ServerHostPort

    cmd = ActiveListener.parse(["--server", "example.com:8080"])

    # Server should be a ServerHostPort with direct access to host and port
    assert cmd.server == ServerHostPort(host="example.com", port=8080)
    assert cmd.server.host == "example.com"
    assert cmd.server.port == 8080
