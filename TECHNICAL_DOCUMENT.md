# Game Ops AI System Technical Document

## Problem Understanding

The Game Ops AI System processes multiplayer match data to support fair operations and decision-making for a gaming event. The core goals are to:

- ingest raw player match records from CSV input
- clean and validate incoming data
- derive player-level features and performance metrics
- predict player skill and assign tiers
- detect suspicious players using hybrid rule-based and machine learning methods
- create fair leaderboards that exclude suspicious players
- build matchmaking groups based on region, device, skill, and network ping
- expose results via a backend API and save CSV output files

The application preserves the original CLI pipeline and evolves it into a FastAPI service while keeping existing ML logic and CSV-based persistence.

## Assumptions

- Input data is provided as a CSV file with columns: `player_id`, `match_id`, `region`, `device`, `ping`, `score`, `kills`, `deaths`, `match_duration_seconds`.
- Missing or invalid numeric data can be handled with median imputation.
- Regions and devices are standardized via simple normalization.
- Existing CSV storage is acceptable; no database is required at this stage.
- The dataset is large enough to support statistical anomaly detection, but not so large that a file-based solution is impossible.
- Matchmaking groups should prioritize same region/device, skill proximity, and ping similarity.

## Data Structure

### Input Data

The raw CSV input structure is:

- `player_id` — unique player identifier
- `match_id` — unique match identifier
- `region` — player region string
- `device` — player device string
- `ping` — network latency value
- `score` — match score
- `kills` — kills achieved in the match
- `deaths` — deaths count in the match
- `match_duration_seconds` — match duration in seconds

### Derived Player Statistics

The pipeline aggregates match rows into player-level statistics including:

- `average_score`
- `average_kills`
- `average_deaths`
- `average_ping`
- `average_kd_ratio`
- `average_score_per_min`
- `average_kills_per_min`
- `total_matches`

These metrics are produced by computing match-level metrics first and then grouping by `player_id`.

### CSV Output Files

The service writes the following CSV files under `outputs/`:

- `leaderboard.csv`
- `suspicious_players.csv`
- `matchmaking.csv`

Each file is designed to be consumable by other systems or analytics workflows.

## Leaderboard Logic

The leaderboard is generated from eligible players after suspicious players are removed. The ranking order is:

1. `skill_score` descending
2. `average_kd_ratio` descending
3. `average_deaths` ascending
4. `average_ping` ascending

Players receive both a `global_rank` and `region_rank`.

This approach balances overall player skill with combat effectiveness and connection quality. It uses the normalized weighted skill score as the primary sorting key.

## Suspicious Player Detection Logic

The detection pipeline uses a hybrid model:

### Rule-Based Detection

Players are flagged if they match any of these thresholds at the match level:

- `kills_per_min > 10`
- `score_per_min > 5000`
- `kd_ratio > 25`
- `kills > 100`
- `match_duration_seconds < 30`

A rule score is calculated from how many suspicious categories a player matches.

### Machine Learning Detection

An `IsolationForest` model detects outliers using these aggregated player features:

- `average_score_per_min`
- `average_kills_per_min`
- `average_kd_ratio`
- `average_ping`

A normalized anomaly score is derived from the model output.

### Final Suspicion Score

The final suspicion score uses a weighted blend:

- `rule_score * 0.4`
- `statistical_score * 0.3`
- `ml_score * 0.3`

Players with `suspicion_score >= 0.6` are marked as suspicious and excluded from the leaderboard and matchmaking output.

## Matchmaking Logic

Matchmaking groups are formed with the following rules:

- Same `region`
- Same `device`
- Similar `skill_score`
- Similar `average_ping`

The pipeline uses `KMeans` clustering on `[skill_score, average_ping]` within each region/device group. The number of clusters is roughly equal to `group_size = 10`, capped by the number of unique points in the group.

A `fairness_score` is then computed as:

- `1 - normalized(skill_score_std_dev)` for each cluster

This encourages groups with tighter skill distribution.

## Architecture Explanation

### Layered Design

The backend is structured in layers:

- `app/api/` — FastAPI route handlers
- `app/services/` — reusable orchestration and logic wrappers
- `app/schemas/` — Pydantic request/response models
- `src/` — existing ML and preprocessing modules
- `outputs/` — CSV export storage
- `app/main.py` — FastAPI application entry point

### Flow

1. Client calls `POST /pipeline/run`
2. API layer invokes service pipeline orchestration
3. Service layer calls existing preprocessing, feature engineering, detection, leaderboard, and matchmaking modules
4. Outputs are saved to CSV
5. API endpoints `GET /leaderboard`, `GET /suspicious-players`, and `GET /matchmaking` read saved CSVs and return typed responses

### Deployment

Run with:

```bash
uvicorn app.main:app --reload
```

Swagger docs are exposed at:

```text
http://localhost:8000/docs
```

## Scaling Plan

### Short-term improvements

- Move CSV storage to a database such as PostgreSQL for faster lookup and persistence.
- Add pagination for leaderboard and matchmaking endpoints.
- Limit input size and use chunked CSV ingestion.

### Medium-term improvements

- Cache leaderboard and matchmaking results in memory or Redis to reduce repeated file reads.
- Add async file I/O and non-blocking request handling.
- Convert the pipeline to support incremental updates rather than full batch reruns.

### Long-term improvements

- Add real-time event streaming for live match ingestion.
- Introduce a model retraining pipeline for evolving player behavior.
- Add season-based rank history and stateful player progression.
- Separate services into dedicated leaderboard, anomaly, and matchmaking microservices if traffic demands it.

## Limitations

- Current storage is file-based and not optimized for high concurrency.
- `KMeans` clustering can be sensitive to duplicate feature values and group size assumptions.
- Suspicion logic relies on static thresholds and a single IsolationForest model.
- No authentication or access control is implemented.
- The current dataset is synthetic or CSV-based, not a production event stream.
- The API reads CSV files on each request, which can be slow under heavy load.

## Summary

The refactor preserves the existing ML pipeline and migrates the application to a backend service with a clean, layered architecture. It retains the same business logic while enabling HTTP access, typed responses, and API-driven execution.
