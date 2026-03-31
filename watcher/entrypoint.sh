#!/bin/sh
set -eu

CONFIG="${CONFIG:-/data/nkhajehn/watcher-mcp-server/haiku.rag.yaml}"
DB="${DB:-/data/nkhajehn/watcher-mcp-server/data/haiku_mxbai.rag.lancedb}"

if [ ! -d "$DB" ]; then
  echo "Initializing DB at $DB..."
  haiku-rag --config "$CONFIG" init --db "$DB"
else
  echo "DB already exists at $DB"
fi

python - <<'PY'
import time, urllib.request
url="http://localhost:5001/health"
for _ in range(180):
    try:
        urllib.request.urlopen(url, timeout=2).read()
        print("docling-serve ready")
        break
    except Exception:
        print("docling serve not ready")
        time.sleep(1)
else:
    raise SystemExit("docling-serve not ready")
PY
if [ "$#" -gt 0 ]; then
  exec "$@"
else
  echo "Starting watcher.py..."
  exec python -u watcher.py
fi
#python /app/custom_ingest.py
#exec haiku-rag --config "$CONFIG" serve --monitor --db "$DB"