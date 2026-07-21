 Football Player Performance Analytics System

An end-to-end football analytics platform built with Python,
StatsBomb Open Data, Streamlit, Plotly and Machine Learning.

Features
---------
 Interactive Dashboard
 Player Comparison
 Team Performance
 Tactical Analysis
 Radar Charts
 xG Analysis
 Advanced Statistics
 Data Pipeline

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


## Architecture

```
StatsBombLoader → League/Team/Player → PerformanceAnalyzer → VisualisationFactory → Streamlit Dashboard
```



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
