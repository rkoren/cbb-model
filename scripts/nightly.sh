#!/bin/bash
# Stage-3 nightly handicapper job — run each morning during the CBB season.
# 1) pull last night's results (ESPN → live log), 2) weekly: refresh adjself, 3) build today's board.
# Schedule via launchd (see the plist snippet in scripts/README-nightly.md) or `crontab -e`.
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
set -a; [ -f .env ] && . ./.env; set +a     # export .env (launchd doesn't inherit your shell env) → KENPOM_API_KEY
PY=.venv/bin/python
LOGDIR=monitoring/nightly-logs; mkdir -p "$LOGDIR"
TODAY=$(date +%F)
exec >>"$LOGDIR/$TODAY.log" 2>&1
echo "=== nightly run $(date) ==="

SEASON=$(date +%Y); [ "$(date +%m | sed 's/^0//')" -ge 8 ] && SEASON=$((SEASON + 1))

# 1) Ingest last night's final results into the live log (idempotent; safe to re-run).
$PY scripts/fetch_results.py "$(date -v-1d +%F 2>/dev/null || date -d 'yesterday' +%F)"

# 2) Pull today's KenPom FanMatch (the slate + KenPom's predictions; skip-if-cached).
$PY scripts/fetch_kenpom_fanmatch.py "$SEASON"

# 3) Weekly (Mondays): recompute the adjself as-of snapshots including live games (~5 min).
[ "$(date +%u)" = "1" ] && $PY scripts/refresh_asof.py

# 4) Build today's us-vs-FanMatch board → monitoring/slate_$TODAY.html, then open it.
$PY scripts/daily_slate.py "$TODAY"
open "monitoring/slate_$TODAY.html" 2>/dev/null || true
echo "=== done $(date) ==="
