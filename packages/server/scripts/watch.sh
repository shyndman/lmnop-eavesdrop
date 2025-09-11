#!/bin/bash

# Get script directory and packages directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGES_DIR="$(realpath "$SCRIPT_DIR/../..")"
BUILD_SCRIPT="$SCRIPT_DIR/build.sh"

# Pass through all arguments to build.sh
BUILD_ARGS="$@"

# Check if watchexec is available
if ! command -v watchexec &> /dev/null; then
    echo "Error: watchexec is required but not installed."
    echo "Install with: cargo install --locked watchexec-cli"
    echo "Or see: https://github.com/watchexec/watchexec#installation"
    exit 1
fi

echo "Starting build watcher with args: $BUILD_ARGS"
echo "Watching directories: $PACKAGES_DIR/wire/src, $PACKAGES_DIR/server/src"
echo "Press Ctrl+C to stop watching"

# Change to packages directory for build script context
cd "$PACKAGES_DIR"

# Use watchexec to watch for changes and run the build script
exec watchexec \
    --watch wire/src \
    --watch server/src \
    --ignore '**/__pycache__/**' \
    --ignore '**/.pytest_cache/**' \
    --ignore '**/.venv/**' \
    --ignore '**/venv/**' \
    --ignore '**/.git/**' \
    -- "$BUILD_SCRIPT" $BUILD_ARGS