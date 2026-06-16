# Game Ops AI System

A Python CLI application for processing multiplayer match data, generating fair leaderboards, detecting suspicious players, predicting skill tiers, and creating balanced matchmaking groups.

## Project Overview

This project focuses on data processing and machine learning for multiplayer game operations. It includes:

- Data preprocessing and validation
- Match-level feature engineering
- Player skill score prediction and tier assignment
- Hybrid anomaly detection for suspicious players
- Fair leaderboard generation and ranking
- Matchmaking grouping using clustering
- Interactive command line interface with formatted output

## Folder Structure

```
game_ops_ai/

├── data/
│   └── sample_matches.csv
│
├── outputs/
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

Follow the interactive menu to run the complete pipeline, view the top leaderboard, review suspicious players, or inspect matchmaking groups.

## Example Commands

- Run the complete pipeline: select `1`
- View the top 10 leaderboard: select `2`
- View suspicious player report: select `3`
- View matchmaking groups: select `4`

## Sample Outputs

- `outputs/suspicious_players.csv`
- `outputs/leaderboard.csv`
- `outputs/matchmaking.csv`

{
  "input_path": "app/data/sample_matches.csv",
  "output_path": "outputs"
}

Each file is generated automatically after the pipeline runs.

## Assumptions

- The dataset uses standard fields for match-level performance.
- Synthetic data is generated automatically when `data/sample_matches.csv` is missing.
- Suspicious behavior is flagged with both rule-based thresholds and anomaly detection.
- Matchmaking is grouped by region and device first, then clustered by skill and ping.

## Future Improvements

- Real-time event streaming from live game servers
- Redis leaderboard caching for low-latency ranking
- Model retraining with evolving player behavior
- Season-based leaderboards and progression systems
- REST APIs for integration with game services
