🏆 Football Player Performance Analytics System

An end-to-end football analytics platform built with Python,
StatsBomb Open Data, Streamlit, Plotly and Machine Learning.

Features
---------
✓ Interactive Dashboard
✓ Player Comparison
✓ Team Performance
✓ Tactical Analysis
✓ Radar Charts
✓ xG Analysis
✓ Advanced Statistics
✓ Data Pipeline

Tech Stack
----------
Python
Pandas
NumPy
Plotly
Streamlit
Scikit-learn
StatsBomb Open Data

Architecture
------------
Data Source
      ↓
Data Loader
      ↓
Data Cleaning
      ↓
Analytics Engine
      ↓
Visualization Layer
      ↓
Interactive Dashboard


## Quick Start

```bash
pip install -r requirements.txt
python main.py --demo
streamlit run dashboards/app.py
```

## Run Tests (165 tests, fully offline)

```bash
pytest tests/ -v
```

## Architecture

```
StatsBombLoader → League/Team/Player → PerformanceAnalyzer → VisualisationFactory → Streamlit Dashboard
```

## OOP Patterns Used

| Pattern | Where |
|---|---|
| Abstract Base Class | Player → Outfielder, Goalkeeper |
| Composition | League HAS Teams HAS Players |
| Factory | create_player(), VisualisationFactory |
| Facade | AdvancedAnalyzer wraps Phase 4 engines |
| Singleton | cfg config; AppState session manager |
| Strategy | PositionalClusterer(algorithm=...) |
| Template Method | BaseChart infrastructure |
| Dependency Injection | PerformanceAnalyzer(league) |

## Data

StatsBomb open data — La Liga 2021/22 is richest (competition_id=11, season_id=90).

## CLI

```bash
python main.py --demo                          # 10 matches, fast
python main.py --competition laliga            # full season
python main.py --demo --player "Messi"         # player deep-dive
python main.py --demo --compare "Messi" "Ronaldo"  # comparison
python main.py --demo --save-charts            # write PNGs to reports/
```
