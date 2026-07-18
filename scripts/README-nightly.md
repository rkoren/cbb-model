# Nightly handicapper (Stage 3)

Keeps the morning "us vs FanMatch" board current during the CBB season, on your Mac.

## Pieces
- `fetch_results.py [date]` — ESPN → last night's final men's D1 box scores → `data/live/mreg_live_{season}.csv` (the live results log). Idempotent.
- `fetch_kenpom_fanmatch.py {season}` — today's KenPom FanMatch (the slate + KenPom's predictions).
- `refresh_asof.py` — recompute `adjself_asof.parquet` from Kaggle + live logs (run **weekly**; Elo/box-score ratings need no refresh — `daily_slate` rebuilds them on the fly from the raw log).
- `daily_slate.py [date]` — build the board → `monitoring/slate_<date>.html`.
- `nightly.sh` — orchestrates the above (results → fanmatch → weekly refresh → board → open).

## What refreshes when
| Signal | How it stays current | Cadence |
|---|---|---|
| **Elo, rolling box-score** | recomputed on the fly by `daily_slate` from Kaggle + live log | every run (free) |
| **adjself** (opponent-adjusted) | `refresh_asof.py` recomputes the snapshots | **weekly** (daily is a proven wash) |
| **prior-season priors (`adj_eff`)** | prev-season, static in-season | once at season start |

## Schedule it (launchd)
Make the runner executable, then install a LaunchAgent that fires each morning (e.g. 7:30am):

```bash
chmod +x scripts/nightly.sh
```

`~/Library/LaunchAgents/com.cbb.nightly.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.cbb.nightly</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>/Users/reillykoren/cbb-model/scripts/nightly.sh</string></array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>7</integer><key>Minute</key><integer>30</integer></dict>
  <key>StandardErrorPath</key><string>/Users/reillykoren/cbb-model/monitoring/nightly-logs/launchd.err</string>
  <key>RunAtLoad</key><false/>
</dict></plist>
```
```bash
launchctl load ~/Library/LaunchAgents/com.cbb.nightly.plist    # install
launchctl start com.cbb.nightly                                # test-fire now
```
Requires `KENPOM_API_KEY` in the environment (or `.env`). Per-run logs land in `monitoring/nightly-logs/`.

## First live-season setup (Nov 2026)
1. Backfill the season so far: `for d in <past dates>; do python scripts/fetch_results.py $d; done`
2. `python scripts/refresh_asof.py` once to seed the current-season adjself snapshots.
3. Load the LaunchAgent — from then on it self-updates each morning.
