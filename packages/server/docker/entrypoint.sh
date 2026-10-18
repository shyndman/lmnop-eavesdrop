#!/bin/bash
set -euo pipefail

# All image variants install into this same environment.
exec /opt/venv/bin/python "$@"
