# CLI Interface Contract

## Command Signature

```bash
eavesdrop-active-listener --server HOST:PORT --audio-device DEVICE
```

## Arguments

### Required Arguments
*None - all arguments have defaults*

### Optional Arguments

#### --server
- **Type**: `str`
- **Default**: `"localhost:9090"`
- **Description**: Server address in hostname:port format for eavesdrop transcription server
- **Validation**: Must be valid hostname:port format with port between 1 and 65535
- **Parser**: Custom parser function validates format and extracts components
- **Examples**: `localhost:9090`, `192.168.1.100:8080`, `transcribe.example.com:443`

#### --audio-device
- **Type**: `str`
- **Default**: `"default"`
- **Description**: Name of the audio input device to capture from
- **Validation**: Must be valid audio device name available on system
- **Examples**: `default`, `pulse`, `hw:0,0`, `USB Audio Device`

## Exit Codes

- **0**: Successful execution and clean shutdown
- **1**: General application error (connection failed, audio device unavailable)
- **2**: Configuration error (invalid arguments, missing dependencies)
- **3**: User interruption (Ctrl+C, SIGTERM)

## Output Behavior

### Normal Operation
- Initial connection status messages
- Real-time transcription feedback (optional verbose mode)
- Clean shutdown messages

### Error Conditions
- Connection errors with retry information
- Audio device availability issues
- ydotool system dependency problems

### User Interaction
- Graceful handling of Ctrl+C (SIGINT)
- Connection status updates during operation
- Clear error messages with actionable information

## Environment Variables

### Optional Configuration
- `EAVESDROP_SERVER`: Override default server (hostname:port format)
- `EAVESDROP_AUDIO_DEVICE`: Override default audio device
- `EAVESDROP_LOG_LEVEL`: Set logging verbosity (DEBUG, INFO, WARN, ERROR)

## System Dependencies

### Required
- `ydotool` system package must be installed and accessible
- Audio system with accessible input devices
- Network connectivity to eavesdrop server

### Runtime Validation
- Verify ydotool availability on startup
- Validate audio device accessibility
- Test server connectivity before starting transcription

## Error Handling Contract

### Connection Failures
- Display clear error message with server details
- Attempt automatic reconnection with exponential backoff
- Maximum 5 retry attempts before giving up
- User notification of reconnection attempts

### Audio Device Issues
- Validate device exists and is accessible on startup
- Clear error message if device becomes unavailable during operation
- Graceful shutdown if device cannot be recovered

### System Integration Failures
- ydotool unavailable: Clear installation instructions
- Permission issues: Guidance on required system permissions
- Desktop environment compatibility warnings

## Performance Expectations

### Response Time
- Transcription typing latency: <100ms from message receipt
- Connection establishment: <5 seconds
- Audio streaming startup: <2 seconds

### Resource Usage
- Memory: <50MB during normal operation
- CPU: Minimal impact during idle transcription
- Network: Efficient WebSocket usage with keep-alive

### Reliability
- Handle rapid transcription updates without dropping content
- Maintain text consistency during connection interruptions
- Graceful recovery from temporary system issues