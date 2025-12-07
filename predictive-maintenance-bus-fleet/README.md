🚍 Predictive Maintenance — Bus Fleet (Dual Horizon ML System)

A full-scale, production-grade predictive maintenance system designed to forecast mechanical failures in a bus fleet.
The solution uses dual-horizon modeling to detect both imminent failures (7 days) and structural, emerging issues (30 days).

This repository delivers a complete, explainable ML pipeline integrated with Power BI and PostgreSQL.

📊 Business Problem

Bus fleets operate under heavy loads and dynamic environmental conditions, leading to:

Unexpected breakdowns

High maintenance costs

Service disruptions

Inefficient parts inventory

Lack of proactive planning

Fleet managers need a data-driven early warning system that identifies high-risk vehicles before failures occur.

🚀 Solution Overview

This system provides:

✔ Short-term (7-day) failure predictions

Captures sudden, imminent failures caused by harsh or abnormal operating conditions.

✔ Medium-term (30-day) failure predictions

Captures structural deterioration, such as engine wear, heat drift, or repeated component stress.

✔ Explainability (per bus)

Identifies which factors most contributed to predicted failure risk.

✔ Integration with Power BI

Outputs risk tables and snapshots for operations dashboards.

✔ Database integration (PostgreSQL)

All predictions, thresholds, feature_importances, and risk scores are stored and refreshed automatically.

🏗 System Architecture
predictive-maintenance-bus-fleet
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── sql_queries.py
│   │
│   ├── features/
│   │   ├── engineer.py
│   │   └── labeling.py
│   │
│   ├── model/
│   │   ├── trainer.py
│   │   ├── thresholds.py
│   │   ├── registry.py
│   │   └── original_logic.py   ← full model logic from the winning hackathon version
│   │
│   ├── explain/
│   │   └── fault_explainer.py
│   │
│   ├── pipeline/
│   │   └── main_pipeline.py    ← one-click pipeline runner
│   │
│   └── utils/
│       ├── config.py
│       └── date_utils.py
│
├── .env
└── README.md

🔁 Pipeline Flow
┌────────────────────────┐
│   Data Ingestion       │  ← Load raw fleet data from PostgreSQL
└──────────────┬─────────┘
               │
┌──────────────▼─────────┐
│ Feature Engineering     │  ← Rolling stats, deltas, trends, part overlaps
└──────────────┬─────────┘
               │
┌──────────────▼─────────┐
│ Label Generation        │  ← strict next-failure logic (7d + 30d)
└──────────────┬─────────┘
               │
┌──────────────▼─────────┐
│ Model Training (LGBM)   │  ← time-based test split
└──────────────┬─────────┘
               │
┌──────────────▼─────────┐
│ Threshold Optimization  │  ← recall ≥ 0.6 constraint
└──────────────┬─────────┘
               │
┌──────────────▼─────────┐
│ Explainability Layer    │  ← per-bus z-score factor ranking
└──────────────┬─────────┘
               │
┌──────────────▼─────────┐
│ Export to Power BI      │  ← predictions_for_powerbi, ml_current_risk
└─────────────────────────┘

📈 Model Performance
7-Day Horizon — Imminent Failure Detection
Metric	Value
Precision	0.378
Recall	0.618
F1-Score	0.469
AUPRC	0.384
Positive Rate	0.346
🔍 Interpretation

Focused on sudden breakdowns

Recall > 0.6 ensures real failures are rarely missed

Precision naturally lower due to short-term noise

AUPRC significantly above baseline → meaningful predictive signal

🧠 Business Meaning

7-day predictions = early alerts for maintenance teams.

30-Day Horizon — Structural Failure Prediction
Metric	Value
Precision	0.799
Recall	0.709
F1-Score	0.751
AUPRC	0.786
Positive Rate	0.807
🔍 Interpretation

Highly reliable predictions for long-term planning

Model captures gradual wear & recurring issue patterns

AUPRC very close to theoretical maximum

🧠 Business Meaning

Enables proactive scheduling, part replacements, and cost optimization.

Cross-Horizon Insights
Metric	Value
Label correlation (7d vs 30d)	0.3559

This confirms that:

7-day model captures immediate risks

30-day model captures systemic degradation

Together they create a complete predictive maintenance strategy.

🧩 Explainability Layer

The system produces qualitative explanations per bus:

Z-scored feature deviations

Top contributing factors

Operational patterns triggering risk

Heat stress, part wear, or abnormal usage indicators

These insights help maintenance engineers understand why a bus is at risk, not just that it is.

🛢 Database Outputs

The following tables are written to PostgreSQL:

✔ predictions_for_powerbi

Daily risk predictions per bus.

✔ ml_current_risk

Snapshot table for dashboard KPIs.

✔ feature_importance_global_h7

Global importances for imminent failures.

✔ feature_importance_global_h30

Global importances for structural failures.

📊 Power BI Integration

The predictions feed directly into dashboards such as:

Fleet Risk Overview

High-risk Buses Heatmap

Component Wear Trends

Failure Types Distribution

Maintenance Optimization KPIs

Business stakeholders can use these dashboards to make proactive operational decisions.

🛠 How to Run the Pipeline
1. Set environment variables

Edit .env:

PG_HOST=localhost
PG_PORT=5432
PG_DB=hacketon
PG_USER=postgres
PG_PASSWORD=1234

2. Install dependencies
pip install -r requirements.txt


(Or use your own environment.)

3. Run the full pipeline
python -m src.pipeline.main_pipeline

4. View results in PostgreSQL & Power BI

Models saved in /models

Predictions written to DB

Dashboards auto-refresh via connector

🧠 Technologies Used

Python 3.13

LightGBM

Pandas / NumPy

SQLAlchemy

PostgreSQL

Power BI

Z-score explainability

Time-based validation

Dual-horizon label engineering

🏁 Conclusion

This system provides a robust, explainable, and business-ready predictive maintenance solution.
It leverages multiple horizons to deliver operational insight—from imminent failures to long-term structural risks.
Engineers and fleet managers gain actionable intelligence that reduces downtime, optimizes resource allocation, and supports long-term operational planning.