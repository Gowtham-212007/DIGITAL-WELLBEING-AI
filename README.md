# 🧠 Digital Well-Being AI — Behavioral Analytics & Addiction Risk Platform

An end-to-end Machine Learning and Behavioral Analytics platform that evaluates daily smartphone habits, predicts digital addiction risk using a **200-tree Random Forest pipeline**, segments users into **4 empirical behavioral archetypes via K-Means clustering**, and provides an interactive **What-If Habit Simulator** calibrated on a 7,500-user dataset.

---

## 📂 Organized Project Architecture

```text
DIGITAL-WELLBEING-AI-main/
├── README.md                  # Project documentation, architecture & quickstart guide
├── requirements.txt           # Python dependencies
├── app/                       # Modular application package
│   ├── __init__.py
│   ├── app.py                 # Main Streamlit dashboard entry point
│   ├── config.py              # Paths, 7,500-user cohort constants, cluster metadata & presets
│   ├── engine.py              # Pure ML inference, K-Means clustering, validation & simulator math
│   ├── styles.py              # Dark glassmorphism CSS & Plotly theme configuration
│   └── digital_wellbeing_ai.html # Standalone single-file interactive web dashboard
├── data/                      # Empirical dataset (7,500 users, 16 features)
│   └── Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
├── models/                    # Serialized scikit-learn pipelines & clustering artifacts
│   ├── random_forest_addiction_model.pkl
│   ├── model_features.pkl
│   ├── kmeans_behavioral_model.pkl
│   └── clustering_scaler.pkl
├── notebook/                  # Jupyter notebooks for EDA, ML training & clustering analysis
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Wellbeing_Analysis.ipynb
│   └── behavioral_clusters.csv
└── tests/                     # Automated unit & integration test suite
    ├── __init__.py
    └── test_app.py
```

---

## ✨ Key Capabilities & UI Organization

1. **Executive Summary & 24-Hour Time Budget Bar**
   - Live KPIs (`Wellness Score`, `Composite Risk Score`, `ML Addiction Probability`, `K-Means Cluster`, `Daily Screen Time`, `Sleep Recovery`) benchmarked against the 7,500-user cohort.
   - Real-time 24-hour time allocation meter partitioning Social Media, Gaming, Productive Screen, Sleep, and Unplugged Offline hours.
2. **Step 1 — Profile & Behavioral Assessment**
   - Structured 3-zone input form (`Demographics & Exposure`, `Activity Breakdown`, `Engagement & Stress`) with physical constraint validation (`Social + Gaming ≤ Screen Time` and `Screen Time + Sleep ≤ 24h`).
   - 4 One-Click Persona Presets (`Balanced Professional`, `Heavy Social Scroller`, `Competitive Gamer`, `Moderate Notification-Busy`) mapped to the 4 K-Means clusters.
3. **Step 2 — AI Addiction Risk & K-Means Cluster Diagnostics**
   - **Random Forest Classifier**: 93.73% test accuracy (0.9887 ROC-AUC) with proper `OneHotEncoder` categorical gender handling.
   - **K-Means Behavioral Clustering (k=4)**: Standardized 8-feature Euclidean distance matching across `Cluster 0 (Balanced)`, `Cluster 1 (Gaming-Heavy)`, `Cluster 2 (Moderate/Notification-Driven)`, and `Cluster 3 (Screen & Social-Heavy)`.
4. **Step 3 — Interactive What-If Simulator & Prioritized Action Plan**
   - 5 real-time habit sliders projecting a 7-day adaptation curve and generating prioritized (`HIGH`, `MEDIUM`, `OPTIMAL`) recommendations.
5. **Step 4 — 7,500-User Cohort Benchmark Explorer**
   - Side-by-side comparison of your active profile against all 4 empirical behavioral clusters.

---

## 🚀 Quickstart & Testing

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit application
streamlit run app/app.py

# 3. Run the automated unit & integration test suite
python3 -m unittest discover -s tests -v
```
