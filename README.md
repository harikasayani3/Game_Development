# Game Ops AI System

A Python application for processing multiplayer match data, generating fair leaderboards, detecting suspicious players, predicting skill tiers, and creating balanced matchmaking groups. Provides both a CLI and REST API interface.

## Project Overview

This project focuses on data processing and machine learning for multiplayer game operations. It includes:

- Data preprocessing and validation
- Match-level feature engineering
- Player skill score prediction and tier assignment
- Hybrid anomaly detection for suspicious players
- Fair leaderboard generation and ranking
- Matchmaking grouping using clustering
- Interactive command line interface with formatted output
- FastAPI REST backend with Swagger documentation

## Folder Structure

```
game_ops_ai/

├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   ├── api/
│   │   ├── anomaly.py            # Suspicious player detection endpoints
│   │   ├── leaderboard.py        # Leaderboard endpoints
│   │   ├── matchmaking.py        # Matchmaking endpoints
│   │   └── matches.py            # Pipeline execution endpoint
│   ├── schemas/
│   │   ├── requests.py           # Request models
│   │   └── responses.py          # Response models
│   ├── services/
│   │   ├── anomaly_detection.py
│   │   ├── feature_engineering.py
│   │   ├── leaderboard.py
│   │   ├── matchmaking.py
│   │   ├── pipeline.py           # Pipeline orchestration
│   │   ├── preprocess.py
│   │   ├── skill_prediction.py
│   │   └── utils.py
│   └── data/
│       └── sample_matches.csv
│
├── data/
│   └── sample_matches.csv
│
├── outputs/
│   ├── leaderboard.csv
│   ├── matchmaking.csv
│   └── suspicious_players.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── skill_prediction.py
│   ├── anomaly_detection.py
│   ├── leaderboard.py
│   ├── matchmaking.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

1. Create a Python 3.11+ virtual environment:

```bash
python -m venv venv
```

2. Activate the environment:

- Windows PowerShell:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- Windows CMD:
  ```cmd
  .\venv\Scripts\activate.bat
  ```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the CLI

From the project root of the `game_ops_ai` folder:

```bash
cd game_ops_ai
python main.py
```

Follow the interactive menu to:
1. Run the complete pipeline
2. View the top 10 leaderboard
3. Review suspicious player report
4. View matchmaking groups

## Running the API Server

Start the FastAPI backend:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

**Swagger UI Documentation:** `http://localhost:8000/docs`

## API Endpoints

### Pipeline Execution
- **POST** `/pipeline/run`
  - Execute the complete pipeline with optional custom input/output paths
  - Request body:
    ```json
    {
      "input_path": "data/sample_matches.csv",
      "output_path": "outputs"
    }
    ```
  - Returns: `{message, players_processed, suspicious_players, matchmaking_groups}`

### Leaderboard
- **GET** `/leaderboard`
  - Query parameters:
    - `region` (optional): Filter by region
    - `limit` (optional): Number of results (default: 10)
  - Returns: List of leaderboard entries sorted by skill score

### Suspicious Players
- **GET** `/suspicious-players`
  - Returns: List of suspicious players with suspicion scores and reasons
  - Reasons include:
    - `high_kills_per_min` (> 10)
    - `high_score_per_min` (> 5000)
    - `high_kd_ratio` (> 25)
    - `high_kills` (> 100)
    - `short_match_duration` (< 30 seconds)

### Matchmaking
- **GET** `/matchmaking`
  - Returns: List of matchmaking groups with fairness scores

### Health Check
- **GET** `/`
  - Returns: `{status: "running"}`

## Sample Outputs

After running the pipeline, the following files are generated in `outputs/`:

- `suspicious_players.csv` - Players flagged as suspicious with scores and reasons
- `leaderboard.csv` - Ranked leaderboard with skill scores and tiers
- `matchmaking.csv` - Matchmaking group assignments with fairness scores

## Anomaly Detection Logic

The system uses a **hybrid model** to detect suspicious players:

### 1. Rule-Based Detection
Flags matches if:
- `kills_per_min > 10`
- `score_per_min > 5000`
- `kd_ratio > 25`
- `kills > 100`
- `match_duration_seconds < 30`

Rule Score = `min(1.0, number_of_violations / 3.0)`

### 2. Statistical Scoring
Computes suspiciousness using aggregated features:
- `average_score_per_min`
- `average_kills_per_min`
- `average_kd_ratio`
- `average_ping`

Features are normalized and mean is computed → `statistical_score`

### 3. Machine Learning Scoring
`IsolationForest` model detects statistical outliers on the same 4 features → `ml_score`

### 4. Final Suspicion Score
Weighted combination:
$$suspicion\_score = (rule\_score \times 0.4) + (statistical\_score \times 0.3) + (ml\_score \times 0.3)$$

Players with `suspicion_score >= 0.6` are marked as suspicious and excluded from leaderboards and matchmaking.

## Leaderboard Logic

Ranking by:
1. `skill_score` (descending) - Primary metric
2. `average_kd_ratio` (descending) - Combat effectiveness
3. `average_deaths` (ascending) - Lower is better
4. `average_ping` (ascending) - Connection quality

Computed ranks:
- `global_rank` - Overall ranking
- `region_rank` - Regional ranking within each region

## Matchmaking Logic

Groups are formed by:
1. Same `region`
2. Same `device`
3. Similar `skill_score` (using KMeans clustering)
4. Similar `average_ping`

Fairness score is calculated as:
$$fairness\_score = \frac{\alpha}{\sigma_{skill}^2 + \alpha}$$

Where $\alpha = 0.05$ and $\sigma_{skill}^2$ is the skill variance within the group. Higher fairness indicates more balanced groups.

## Architecture

The system uses a **layered design**:

- **API Layer** (`app/api/`) - HTTP endpoints
- **Service Layer** (`app/services/`) - Business logic orchestration
- **Schema Layer** (`app/schemas/`) - Request/response models
- **Core Layer** (`src/`) - ML algorithms and preprocessing

The pipeline flow:
1. Load raw match CSV
2. Preprocess and validate data
3. Compute match-level metrics
4. Aggregate player statistics
5. Predict skill scores and tiers
6. Detect suspicious players
7. Generate leaderboard (excluding suspicious)
8. Create matchmaking groups (excluding suspicious)
9. Save outputs to CSV

## Assumptions

- Input CSV has columns: `player_id`, `match_id`, `region`, `device`, `ping`, `score`, `kills`, `deaths`, `match_duration_seconds`
- Missing numeric values are imputed with median
- Regions and devices are standardized via normalization
- Synthetic data is generated automatically when input file is missing
- Dataset is large enough for statistical anomaly detection

## Future Improvements

- Database integration (PostgreSQL) instead of CSV storage
- Redis caching for leaderboards and matchmaking
- Pagination for large result sets
- Async file I/O and non-blocking request handling
- Incremental pipeline updates instead of full batch reruns
- Real-time event streaming from game servers
- Model retraining pipeline for evolving player behavior
- Season-based rank history and player progression
- Microservices architecture for scaling
- Authentication and access control

