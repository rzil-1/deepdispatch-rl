# 🚔 DeepDispatch RL: Parking Enforcement Intelligence

**DeepDispatch RL** is an AI-driven dispatch prototype built for the GridLock Hackathon. It solves the operational challenge of reactive, patrol-based parking enforcement by combining geospatial analytics with Reinforcement Learning (RL). 

Instead of waiting for traffic jams to happen or blindly patrolling quiet streets, DeepDispatch predicts where high-impact parking violations will occur and dynamically routes officers to clear them *before* they cause gridlock.

---

## 🌟 Key Features

* **Congestion Impact Score (CIS):** A custom mathematical proxy that prioritizes violations based on vehicle weight and road type, ensuring officers target massive trucks on arterial roads instead of scooters in alleys.
* **Geospatial Hotspot Engine:** Uses 6-character Geohashing and Kernel Density Estimation (KDE) to translate 298K historical records into smooth, predictive hotspot maps.
* **Reinforcement Learning Dispatch:** An AI agent (trained via PPO) that acts as a virtual patrol officer in a custom `Gymnasium` simulator. It learns a routing policy that clears **18% more actual congestion** than baseline heuristic patrolling.
* **Interactive Dashboard:** A premium, real-time Streamlit dashboard featuring Folium heatmaps, Plotly metrics, and a deep-dive into the RL agent's performance scorecards.

---

## 📂 Project Structure

```text
DeepDispatch-RL/
├── data/
│   ├── raw/                 # Place your raw CSV dataset here
│   └── processed/           # Auto-generated parquet and pkl files
├── docs/                    # Technical deep dives and non-technical explanations
├── src/                     # Core Python modules
│   ├── data_engineering.py     # Pipeline & CIS logic
│   ├── congestion_intelligence.py  # KDE and probability tables
│   └── rl_environment.py       # Custom MDP Simulator & Baselines
├── scripts/                 # Exploration notebooks/scripts
├── results/                 # Trained AI models and evaluation logs
├── app.py                   # Streamlit Dashboard Entrypoint
├── train.py                 # Master Training Pipeline Entrypoint
└── requirements.txt         # Dependency list
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your machine.

### 2. Create a Virtual Environment
It is highly recommended to isolate the project dependencies.
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Provide the Dataset
*(Note: To keep the repository lightweight and within GitHub limits, the large 100MB data files are intentionally ignored in `.gitignore`.)*

Download the hackathon dataset (`jan to may police violation_anonymized791b166.csv`) and place it in the `data/raw/` directory. If the file has a different name, update the `csv_path` variable in `train.py`.

### 5. Run the Training Pipeline
Before you can view the dashboard, you must run the data engineering and AI training pipeline. This will process the CSV, train the DQN and PPO agents, and save the evaluation results.
```bash
python train.py
```
*(Note: Training 50,000 steps should take about 2-5 minutes depending on your hardware).*

### 6. Launch the Dashboard
Once training is complete and the `processed/` and `results/` folders are populated, launch the interactive UI:
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---
*Built for GridLock Hackathon 2.0*
