#!/bin/sh
set -eu

# The ColQwen index (colrag/store.py) is created on the first indexed PDF; nothing to initialize.
if [ "$#" -gt 0 ]; then
  exec "$@"
else
  echo "Starting watcher.py..."
  exec python -u watcher.py
fi
