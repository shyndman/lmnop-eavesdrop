# Active Listener Quickstart Guide

## Installation and Setup

### 1. System Requirements
```bash
# Install ydotool system dependency (Ubuntu/Debian)
sudo apt install ydotool

# Verify ydotool installation
ydotool --version

# Add user to input group (required for uinput access)
sudo usermod -a -G input $USER
# Log out and back in for group changes to take effect
```

### 2. Project Setup
```bash
# Navigate to project root
cd /path/to/eavesdrop

# Create active-listener package
uv init --name active-listener --app packages/active-listener

# Navigate to new package
cd packages/active-listener

# Install dependencies
uv add eavesdrop-client eavesdrop-wire clypi python-ydotool structlog
```

### 3. Development Dependencies
```bash
# Add development and testing dependencies
uv add --group dev pytest pytest-asyncio pytest-mock structlog
```

## Quick Start Development

### 1. Basic Application Structure
```bash
mkdir -p src/eavesdrop/active_listener
touch src/eavesdrop/active_listener/__init__.py
touch src/eavesdrop/active_listener/__main__.py
touch src/eavesdrop/active_listener/cli.py
touch src/eavesdrop/active_listener/client.py
touch src/eavesdrop/active_listener/text_manager.py
touch src/eavesdrop/active_listener/typer.py
```

### 2. Test Structure Setup
```bash
mkdir -p tests/{integration,unit}
touch tests/__init__.py
touch tests/integration/test_client_connection.py
touch tests/integration/test_end_to_end.py
touch tests/unit/test_text_manager.py
touch tests/unit/test_typer.py
```

## Running the Application

### 1. Basic Usage
```bash
# From packages/active-listener directory
python -m eavesdrop.active_listener

# With custom server
python -m eavesdrop.active_listener --host 192.168.1.100 --port 9090

# With specific audio device
python -m eavesdrop.active_listener --audio-device "USB Audio Device"
```

### 2. Environment Variable Configuration
```bash
# Set default server
export EAVESDROP_HOST=transcribe.example.com
export EAVESDROP_PORT=9090
export EAVESDROP_AUDIO_DEVICE="hw:1,0"

# Run with environment defaults
python -m eavesdrop.active_listener
```

### 3. Debug Mode
```bash
# Enable debug logging
export EAVESDROP_LOG_LEVEL=DEBUG
python -m eavesdrop.active_listener
```

## Development Workflow

### 1. Running Tests
```bash
# Unit tests only
pytest tests/unit/

# Integration tests (requires running eavesdrop server)
pytest tests/integration/

# All tests
pytest
```

### 2. Code Quality
```bash
# Type checking
pyright

# Linting
ruff check

# Formatting
ruff format
```

### 3. Local Testing Setup
```bash
# Terminal 1: Start eavesdrop server
cd ../server
eavesdrop-server --config config.yaml --port 9090

# Terminal 2: Run active listener
cd ../active-listener
python -m eavesdrop.active_listener --host localhost --port 9090
```

## Verification Steps

### 1. System Integration Test
```bash
# Test ydotool availability
ydotool type "test message"

# Verify audio device access
python -c "import sounddevice; print(sounddevice.query_devices())"

# Test eavesdrop server connectivity
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Key: test" -H "Sec-WebSocket-Version: 13" \
  http://localhost:9090/ws
```

### 2. Application Integration Test
```bash
# Open a text editor (gedit, nano, etc.)
gedit &

# Run active listener in another terminal
python -m eavesdrop.active_listener

# Speak into microphone and verify text appears in editor
```

### 3. Error Handling Test
```bash
# Test with invalid server
python -m eavesdrop.active_listener --host invalid-host

# Test with invalid audio device
python -m eavesdrop.active_listener --audio-device "nonexistent"

# Test permission issues (if applicable)
# Remove user from input group and retry
```

## Common Issues and Solutions

### 1. ydotool Not Working
```bash
# Check uinput module
lsmod | grep uinput

# Load uinput if missing
sudo modprobe uinput

# Check device permissions
ls -la /dev/uinput

# Verify group membership
groups $USER | grep input
```

### 2. Audio Device Issues
```bash
# List available audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Test specific device
python -c "import sounddevice; sounddevice.check_input_settings(device='default')"
```

### 3. Connection Issues
```bash
# Test server availability
nc -zv localhost 9090

# Check firewall
sudo ufw status

# Verify server logs
tail -f /path/to/eavesdrop/server/logs/
```

## Performance Testing

### 1. Latency Testing
```bash
# Enable debug logging to measure timing
export EAVESDROP_LOG_LEVEL=DEBUG
python -m eavesdrop.active_listener

# Speak short phrases and observe timestamp logs
# Look for: transcription_received → typing_completed duration
```

### 2. Stress Testing
```bash
# Continuous speech test
# Speak continuously for 2-3 minutes
# Verify no text loss or duplication

# Rapid speech test
# Speak quickly with frequent pauses
# Verify proper segment handling
```

### 3. Resource Monitoring
```bash
# Monitor memory and CPU usage
top -p $(pgrep -f active_listener)

# Monitor network connections
ss -tuln | grep 9090
```

## Troubleshooting Checklist

### Before Reporting Issues
- [ ] ydotool is installed and accessible
- [ ] User is in input group (logout/login required)
- [ ] Eavesdrop server is running and reachable
- [ ] Audio device is accessible and not in use by other applications
- [ ] All Python dependencies are installed correctly
- [ ] System has network connectivity to server

### Debug Information to Collect
```bash
# System information
uname -a
ydotool --version
python --version
groups $USER

# Application logs with debug enabled
export EAVESDROP_LOG_LEVEL=DEBUG
python -m eavesdrop.active_listener 2>&1 | tee debug.log

# Network connectivity
ping -c 3 $EAVESDROP_HOST
telnet $EAVESDROP_HOST $EAVESDROP_PORT
```

### Development Environment Reset
```bash
# Clean virtual environment
uv sync --reinstall

# Reset configuration
unset EAVESDROP_HOST EAVESDROP_PORT EAVESDROP_AUDIO_DEVICE

# Test with minimal configuration
python -m eavesdrop.active_listener --help
```