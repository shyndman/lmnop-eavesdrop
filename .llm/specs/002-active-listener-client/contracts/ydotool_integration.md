# ydotool Integration Contract

## Desktop Typing Interface

### Typing Operations Contract

```python
class DesktopTyper:
    def __init__(self):
        """Initialize desktop typer and pydotool connection."""

    def type_text(self, text: str) -> None:
        """Type text at the current cursor position.

        Args:
            text: Text to type (must be valid UTF-8)

        Raises:
            TypingError: If ydotool is unavailable or typing fails
            ValueError: If text contains characters that cannot be typed

        Behavior:
            - Types text at current desktop cursor position
            - Preserves current application focus
            - Handles Unicode characters correctly
            - Does not modify clipboard or selection
        """

    def delete_characters(self, count: int) -> None:
        """Delete characters by sending backspace keystrokes.

        Args:
            count: Number of characters to delete (must be >= 0)

        Raises:
            TypingError: If ydotool is unavailable or deletion fails
            ValueError: If count < 0

        Behavior:
            - Sends backspace keystrokes to delete characters
            - Deletes from current cursor position backward
            - Handles multi-byte Unicode characters correctly
            - Does not affect other desktop state
        """

    def execute_text_update(self, update: TextUpdate) -> None:
        """Execute complete text update operation atomically.

        Args:
            update: Update containing deletions and text to type

        Raises:
            TypingError: If any part of the operation fails

        Atomicity:
            - Performs deletions first, then typing
            - If typing fails after deletion, attempts recovery
            - Logs all operations for debugging
            - Ensures desktop text remains in consistent state
        """
```

## System Integration Contract

### ydotool Availability

```python
def check_ydotool_available() -> bool:
    """Check if ydotool is installed and accessible.

    Returns:
        True if ydotool can be used for desktop automation

    Checks:
        - ydotool executable exists in PATH
        - Current user has necessary permissions
        - uinput kernel module is available
    """

def get_ydotool_version() -> str:
    """Get version information from ydotool installation.

    Returns:
        Version string from ydotool --version

    Raises:
        SystemError: If ydotool is not available

    Used for:
        - Compatibility verification
        - Debug information in error reports
    """
```

### Permission and Access Control

```python
def verify_typing_permissions() -> None:
    """Verify application has permissions needed for desktop typing.

    Raises:
        PermissionError: If required permissions are missing
        SystemError: If system configuration prevents typing

    Requirements:
        - Access to /dev/uinput device
        - Appropriate user group membership
        - No conflicting desktop security policies
    """
```

## Error Handling Contract

### Typing Error Classification

```python
class TypingError(Exception):
    """Base exception for desktop typing failures."""

class SystemUnavailableError(TypingError):
    """ydotool system is not available or accessible."""

class OperationFailedError(TypingError):
    """Specific typing operation failed."""

class PermissionError(TypingError):
    """Insufficient permissions for desktop automation."""
```

### Recovery Strategies

```python
def attempt_typing_recovery(
    failed_operation: TextUpdate,
    error: TypingError
) -> bool:
    """Attempt to recover from failed typing operation.

    Args:
        failed_operation: The operation that failed
        error: The error that occurred

    Returns:
        True if recovery was successful

    Recovery Strategies:
        - Retry operation after brief delay
        - Check system permissions and guidance
        - Verify ydotool availability
        - Log failure details for user debugging
    """
```

## Performance Contract

### Response Time Requirements

- `type_text()`: Complete within 50ms for typical text lengths (<100 chars)
- `delete_characters()`: Complete within 20ms for reasonable deletion counts (<50 chars)
- `execute_text_update()`: Complete within 100ms for combined operations

### Throughput Requirements

- Handle rapid successive text updates without dropping operations
- Queue management for operations that arrive faster than execution
- Backpressure handling if typing cannot keep up with transcription

### Resource Usage

- Minimal CPU usage during idle periods
- No memory leaks from repeated typing operations
- Efficient ydotool process management

## Desktop Environment Compatibility

### Supported Environments

```python
def get_desktop_compatibility() -> DesktopInfo:
    """Get information about desktop environment compatibility.

    Returns:
        Information about current desktop and ydotool compatibility

    Compatibility Matrix:
        - X11: Full support via ydotool
        - Wayland: Full support via ydotool
        - Console: Not applicable for typing operations
    """
```

### Environment-Specific Behavior

- **Wayland**: Works without additional configuration
- **X11**: Works with standard ydotool installation
- **Remote Desktop**: May have limitations depending on protocol
- **Container Environments**: Requires appropriate device access

## Security and Safety Contract

### Input Sanitization

```python
def sanitize_typing_input(text: str) -> str:
    """Sanitize text input to prevent security issues.

    Args:
        text: Raw text from transcription

    Returns:
        Sanitized text safe for desktop typing

    Sanitization Rules:
        - Remove or escape shell metacharacters
        - Filter potentially harmful Unicode sequences
        - Preserve normal text content and whitespace
        - Log any modifications for debugging
    """
```

### Operation Logging

```python
def log_typing_operation(
    operation: str,
    content_length: int,
    success: bool,
    duration_ms: float
) -> None:
    """Log typing operation for audit and debugging.

    Args:
        operation: Type of operation (type, delete, update)
        content_length: Length of content processed
        success: Whether operation completed successfully
        duration_ms: Time taken for operation

    Privacy:
        - Never logs actual text content
        - Only logs metadata for debugging
        - Respects user privacy while enabling troubleshooting
    """
```

## Testing Contract

### Mock Interface for Testing

```python
class MockDesktopTyper(DesktopTyper):
    """Mock implementation for testing text processing logic.

    Behavior:
        - Records all operations instead of executing them
        - Allows verification of typing sequence
        - Simulates various error conditions for testing
        - Provides access to typed content for assertions
    """

    def get_typed_content(self) -> str:
        """Get the content that would have been typed to desktop."""

    def get_operation_log(self) -> list[TypingOperation]:
        """Get log of all typing operations performed."""

    def simulate_error(self, error_type: type[TypingError]) -> None:
        """Configure mock to simulate specific error conditions."""
```

### Integration Testing

```python
def create_real_typer_for_testing() -> DesktopTyper:
    """Create real typer instance for integration testing.

    Returns:
        Real DesktopTyper that actually performs desktop operations

    Safety:
        - Only used in controlled test environments
        - Includes safeguards against typing in wrong applications
        - Provides cleanup mechanisms after tests
    """
```

## Configuration Contract

### Typing Behavior Configuration

```python
@dataclass
class TypingConfig:
    """Configuration for desktop typing behavior."""

    typing_speed_ms: int = 0  # Delay between keystrokes (0 = as fast as possible)
    retry_attempts: int = 3  # Number of retry attempts for failed operations
    operation_timeout_ms: int = 5000  # Maximum time to wait for operation completion
    batch_size: int = 100  # Maximum characters to type in single operation

    def validate(self) -> None:
        """Validate configuration values are reasonable."""
```

### Runtime Configuration Updates

```python
def update_typing_config(config: TypingConfig) -> None:
    """Update typing behavior configuration at runtime.

    Args:
        config: New configuration to apply

    Effects:
        - Updates typing speed and retry behavior
        - Validates configuration before applying
        - Logs configuration changes for debugging
    """
```