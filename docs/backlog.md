# cbb-model Backlog

Project backlog for the **college basketball model** — the modeling/data work, broken into
digestible pieces to tackle one at a time. Platform gaps (anything not CBB-specific that the
`kitchen` platform should provide) go in `../kitchen-platform/docs/backlog.md`, not here.

## North Star

Build a model that can **handicap any college basketball game** — predict the *exact score*
of each team, from which everything else is derived:

- **margin** (= ScoreA − ScoreB) → point spread, and **win probability** from the margin
  *distribution* (P(margin > 0)), so Brier is a *consequence* of a sharp score model, not a
  separately-fit win/loss head;
- **total** (= ScoreA + ScoreB) → over/under.

Predicting scores forces the model to genuinely assess teams (pace + efficiency), not just
win/loss. We keep optimizing Brier, but compute it from the score predictions. The current
model already regresses `PointDiff` (margin) → calibrates to win prob; the work below grows
that into a full score model, generalizes it from tournament-only to any game, and benchmarks
it against the market.

## How we work

- One digestible item at a time; each should fit in roughly a session.
- Validate every change on the **2026 frozen holdout** (leak-free) — iterate on CV
  (`loto_brier` over ≤2025), check the holdout sparingly.
- Run experiments via the platform: `kitchen run train --variant <name>`, compare with
  `kitchen leaderboard` / `kitchen diff`.

## Status labels

| Label | Meaning |
|---|---|
| `todo` | Ready to pick up. |
| `doing` | In progress. |
| `blocked` | Needs data or a decision. |
| `done` | Implemented, tested, validated. |

---

## M0: Foundation (done)

| ID | Item | Status |
|---|---|---|
| F-001 | Repo on the unified `menu.yaml` / kitchen structure; `features → train → evaluate` run end-to-end (`kitchen menu run`). | done |
| F-002 | Secrets via `kitchen.secrets` manifest (KENPOM_API_KEY ← AWS SM); CI auth via GitHub OIDC. | done |
| F-003 | KenPom efficiency overlay (AdjOE/AdjDE/AdjEM/AdjTempo) merged into men's `adj_eff` from cached ratings. | done |
| F-004 | KenPom **rich** features (`d_kp_SOS/Luck/APL_Off/APL_Def`) joined into `matchups.parquet` (`_apply_kenpom`, additive). | done |
| F-005 | `kenpom_rich` experiment via the platform `variants:` overlay — beat baseline on CV (loto_brier 0.1627 vs 0.1653). | done |
| F-006 | 2026 frozen-holdout scaffold: `data/holdout/` contract, `cbb.holdout` build+score, `holdout_brier` logged at train (gated). Validated on real 2025 data. | done |

## M1: Finish the KenPom + holdout loop (now)

| ID | Item | Status |
|---|---|---|
| KP-001 | **Source 2026 men's tournament results** → `data/holdout/tourney_results_2026.csv` so we get a real `holdout_brier` for baseline vs `kenpom_rich`. **Finding (2026-06-26):** KenPom `fanmatch` returns *predictions only* (`HomePred/VisitorPred/HomeWP/PredTempo`) — **no actual outcomes** — so it can't be the results source. It IS usable for (a) the exact 2026 game list (filter to the 68-team field) and (b) a **KenPom benchmark** to compare our model against (M4 territory). Actual outcomes must come from elsewhere: user-provided results CSV, an external source (sports-reference/ESPN), or a future Kaggle MMM data refresh (2026 results likely not downloadable until the 2027 competition). **Progress (2026-06-27):** `scripts/build_holdout_template.py` pulls the bracket dates from fanmatch, maps names→TeamIDs, filters to the 68-team field, and writes `data/holdout/2026_template.csv` — all **67 games**, with KenPom's predicted scores + WP as a benchmark column and a blank `Winner` column. `cbb.holdout.finalize_template` converts the filled template → the `tourney_results_2026.csv` contract (4 tests). **Filled + validated (2026-06-27):** winners entered; `scripts/validate_holdout_template.py` checks single-elim invariants (Winner∈game, all 68 field teams, no team plays after losing, one undefeated champion = final winner) + flags KP upsets. It caught **2 data-entry errors** — both upsets recorded as the favorite winning, contradicted by the fanmatch-real later rounds (Iowa beat Florida 03-22; Arizona beat Arkansas 03-26). Corrected → bracket consistent, champion = Michigan (beat UConn). Finalized → `data/holdout/tourney_results_2026.csv` (67 games). **KenPom benchmark to beat: `holdout_brier=0.1563`** (its pre-game WP vs actual). **RESULT (2026-06-27):** baseline `holdout_brier=0.1729` (CV 0.1669) · **kenpom_rich `holdout_brier=0.1534`** (CV 0.1627) · KenPom 0.1563. KenPom rich features generalize to the unseen 2026 tournament (−0.0195 vs baseline) and **edge KenPom's own predictions** (0.1534 < 0.1563); baseline alone trails KenPom. Caveat: 67 games — suggestive, not significant; first holdout peek (don't iterate against it). Tagging fix: variant overlay now sets `model.variant` so the run labels `kenpom_rich` (see kitchen CBB-016 follow-up). | done |
| KP-002 | **Add KenPom height features** (`kp_AvgHgt/HgtEff/Exp/Bench/Continuity`). **Done (2026-06-27):** `scripts/fetch_kenpom_height.py` cached all 17 seasons; height flows into `kenpom_rich` (now 9 kp feats). **Result: height is ~neutral** — improved the 67-game holdout (0.1534→0.1504) but was flat/slightly-worse on CV (0.1627→0.1630); the holdout gain is within small-sample noise. Kept in the variant (feature-selection later) but the proven KenPom win remains the ratings-extra features. Tagging fixed (variant overlay sets `model.variant`). | done |
| KP-004 | **KenPom inputs weren't reaching CI** (surfaced wiring KP-002 durability). `kenpom_ratings_*`/`kenpom_height_*` parquets lived in gitignored, untracked `data/processed/` and `_apply_kenpom` only *reads* (never fetches) → CI's `dvc pull` (data/raw only) left them absent → KenPom silently skipped → **CI model was baseline-only**, invisibly. **Done (2026-06-27):** moved the 34 parquets to a dedicated DVC-tracked `data/kenpom/` (input dir like `data/raw`), pointed `_apply_kenpom`/fetch script there, `dvc add data/kenpom` (pointer `data/kenpom.dvc`). Verified features reads from the new location (9 kp cols + holdout built; 48 tests green). **Remaining (you):** `dvc push` (upload to S3 so CI can pull) + commit `data/kenpom.dvc`, `data/.gitignore`, and the code; also commit `data/holdout/tourney_results_2026.csv` (not gitignored, small) so the holdout builds in CI. | doing |
| KP-003 | Improve KenPom→Kaggle name matching (~60 teams/season fall back to manual efficiency) and stop logging the full unmatched list every season (log a count). **Progress (2026-06-27):** added 6 abbreviation overrides the 0.85 fuzzy missed (Kennesaw St., LIU, McNeese, North Dakota St., Saint Louis, Queens) + fixed two bugs — `"Penn"→"Pennsylvania"` pointed at a non-existent Kaggle team (removed; "Penn" now exact-matches) and added current `"Saint Mary's"` alongside the stale `"Saint Mary's (CA)"`. Cleared all 8 unmapped 2026 tournament teams (holdout now complete). **Remaining:** the ~50 non-tournament low-majors (matter for the M3 regular-season model, not the tourney holdout) + the log-a-count change. | doing |

## M2: Score model (margin + total)

| ID | Item | Status |
|---|---|---|
| SC-001 | Add a **total** head alongside the existing margin regression → predict both team scores. Decompose: margin ≈ strength diff + home court, total ≈ combined pace×efficiency (near-independent). | todo |
| SC-002 | Derive **win probability from the margin distribution** (P(margin > 0)) instead of (or compared against) the current leaf calibration — make Brier a consequence of the score model. | todo |
| SC-003 | Extend holdout scoring with `margin_MAE`, `total_MAE`, and calibration of the derived win prob (alongside `holdout_brier`). | todo |
| SC-004 | Fix the in-sample `evaluate` metric (champion currently scored on seasons it trained on → optimistic ~0.122 vs leak-aware LOTO ~0.165); report a leak-free / holdout number. | todo |

## M3: Handicap any game (regular season)

| ID | Item | Status |
|---|---|---|
| GM-001 | Build a **regular-season game-level dataset** (train on all reg-season games, not just ~3k tournament matchups) — the actual "handicap any game" target. Requires venue (`WLoc`) + a home-court adjustment (~3.5 pts). | todo |
| GM-002 | **As-of-date KenPom ratings** via the `archive` endpoint (snapshot at/just-before each game date) so in-season features are leak-free. Use only non-`Final` archive columns. | todo |
| GM-003 | Possessions/pace features estimated from detailed box scores (FGA, TO, OR, FTA) to sharpen the tempo term. | todo |

## M4: True handicapper (vs the market)

| ID | Item | Status |
|---|---|---|
| MK-001 | **Source historical spreads + totals (closing lines).** KenPom/Kaggle don't carry these; our current Vegas data is moneyline (American odds) only. Needed to benchmark + calibrate. **Blocked on a data source** (Kaggle line data / sportsbook archive / odds API). | blocked |
| MK-002 | Benchmark the score model against the closing line: ATS%, total accuracy, and closing-line value (CLV). "Beat the close" is the real bar for sharpness. | todo |
| MK-003 | Calibrate score/spread/total outputs against the market; expose spread + total + moneyline as the handicapper's outputs. | todo |
