# Betting Market Data — Sources & Plan for Model Benchmarking
 
Context for benchmarking the CBB model's spreads, totals, and win probabilities
against sharp closing lines (Pinnacle, Circa, FanDuel). Researched 2026-07-17.
 
## Strategy (TL;DR)
 
Don't chase 25 years. Stitch two datasets:
 
1. **SBRO archives (2007-08 → 2021-22)** — free consensus Vegas open/close for
   spreads, totals, moneylines, 2H lines.
2. **The Odds API historical (Nov 2020 → present)** — book-specific closing
   snapshots for Pinnacle (`eu` region) and FanDuel (`us` region).
The ~2-season overlap (2020-21, 2021-22) lets us validate consensus close vs.
Pinnacle close directly. This yields ~18 seasons / 90k+ games — plenty of
statistical power. Book-specific lines before 2007 effectively don't exist
publicly (Pinnacle left the US market in 2007; Circa opened late 2019).
 
## Data sources
 
### Tier 1 — Sportsbook Reviews Online (free, frozen archive)
- URL: https://sportsbookreviewsonline.com/scoresoddsarchives/ncaabasketball/ncaabasketballoddsarchives.htm
- Seasons 2007-08 through 2020-21 as .xlsx downloads; 2021-22 as a webpage.
- Fields: opening/closing spread, total, moneyline, 2nd-half lines, scores.
- Known issues: "Vegas consensus" not book-specific; team-name
  inconsistencies; occasional swapped/garbled rows. Requires cleaning pass
  and a team-name crosswalk (Kaggle `MTeamSpellings.csv` helps).
- Archive is explicitly no longer updated. Mirrors exist on Kaggle/GitHub.
### Tier 2 — The Odds API (paid, book-specific, 2020-11-16 →)
- Historical endpoint: https://the-odds-api.com/historical-odds-data/
- NCAAB snapshots from 2020-11-16, at 10-min intervals (5-min after Sep 2022).
- Bookmakers: FanDuel = key `fanduel`, region `us`; Pinnacle = key `pinnacle`,
  region `eu` (note: Pinnacle odds scraped from public site, may lag slightly).
  Circa is NOT covered.
- Closing line = last snapshot before tip. Cost: 10 quota units per region per
  market per snapshot — pull ONE snapshot at tip time per game, not full
  movement history. ~5,500 games/season → affordable on paid tiers.
- Alternative for Pinnacle only: https://api.bettingiscool.com/ (history back
  to 2021, spreads/totals/devigged, €49+/mo — confirm NCAAB coverage first).
### Tier 3 — Circa (capture-it-yourself, going forward only)
- No public API (https://sportsapis.dev/circa-api); only carried by pricier
  aggregators (Unabated, OpticOdds). No retroactive Circa closing-line file
  exists for CBB at consumer prices.
- Plan: scheduled job that snapshots Circa CBB lines at tip time daily during
  the season. Starting Nov 2026 gives a full season of Circa closes by March.
### Pre-2007 (only if ever needed)
- Consensus-only, scattered: Computer Sports World / Don Best archives (paid,
  back to early '90s), covers.com game pages (scrape, back to ~mid-2000s),
  ThePredictionTracker archives (https://www.thepredictiontracker.com/).
- Low marginal value for benchmarking; skip unless doing era studies.
## Benchmarking methodology notes
 
- **Win probabilities:** de-vig the closing moneyline (e.g., proportional or
  Shin method) to get the market's implied probability; compare model prob
  via log loss / Brier vs. market.
- **Spreads/totals:** compare model number to closing number (MAE vs. close);
  track whether model disagreements beat the close (CLV proxy).
- **Consensus vs. Pinnacle:** use the 2020-2022 overlap to measure how far
  consensus close deviates from Pinnacle close before treating pre-2020
  consensus as the "sharp" benchmark.
## Implementation checklist
 
- [ ] Download + clean SBRO season files into one tidy CSV/parquet
      (one row per game: date, teams, open/close spread, total, ML, scores).
- [ ] Build team-name crosswalk: SBRO names ↔ Kaggle TeamIDs ↔ Odds API names.
- [ ] Odds API puller: for each historical game, fetch last pre-tip snapshot
      for spreads/totals/h2h from `us` (FanDuel) + `eu` (Pinnacle); needs API
      key + paid plan.
- [ ] De-vig utility for moneylines → market win probs.
- [ ] Forward-capture job for Circa (and Pinnacle/FanDuel live) at tip time.
- [ ] Validation: consensus-vs-Pinnacle deviation study on overlap seasons.