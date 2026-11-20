"""
Configuration, File Paths, Empirical Cohort Benchmarks, and Persona Presets
for Digital Well-Being AI.
"""

from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
NOTEBOOK_DIR = BASE_DIR / "notebook"

DATA_PATH = DATA_DIR / "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
CLUSTERS_CSV_PATH = NOTEBOOK_DIR / "behavioral_clusters.csv"

RF_MODEL_PATH = MODEL_DIR / "random_forest_addiction_model.pkl"
FEATURES_PATH = MODEL_DIR / "model_features.pkl"
KMEANS_MODEL_PATH = MODEL_DIR / "kmeans_behavioral_model.pkl"
SCALER_PATH = MODEL_DIR / "clustering_scaler.pkl"

MODEL_FEATURES_DEFAULT: List[str] = [
    "age",
    "gender",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]

CLUSTER_FEATURES: List[str] = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]

# Exact StandardScaler parameters from the 7,500-user training cohort
SCALER_MEANS = np.array([7.499912, 3.273484, 2.014183, 3.242420, 6.737561, 134.257333, 97.832400, 9.243827])
SCALER_STDS = np.array([2.609014, 1.585236, 1.145963, 1.600658, 1.283519, 66.582443, 48.420120, 2.718100])

# Exact K-Means (k=4) scaled centroids from kmeans_behavioral_model.pkl
KMEANS_SCALED_CENTERS = np.array([
    [-0.8287, -0.0267, 0.0283, 0.8841, -0.1571, -0.1304, -0.1613, -0.8217],  # Cluster 0
    [0.8383, -0.0191, 0.8842, -0.0427, -0.0611, 0.0130, -0.0283, 0.8197],   # Cluster 1
    [-0.8859, -0.0296, -0.0287, -0.9006, 0.1243, 0.1429, 0.0788, -0.8719],  # Cluster 2
    [0.8467, 0.0770, -0.9336, 0.0346, 0.1020, -0.0221, 0.1168, 0.8452],     # Cluster 3
])

CLUSTER_METADATA: Dict[int, Dict[str, Any]] = {
    0: {
        "name": "Balanced & Work-Focused",
        "badge": "🟢 Balanced Productivity",
        "addiction_rate": 46.37,
        "cohort_count": 1887,
        "color": "#10B981",
        "description": (
            "Characterized by controlled weekday screen time (~5.3h), strong work/study engagement "
            "(~4.7h/day), and moderate weekend usage (~7.0h)."
        ),
        "averages": {
            "daily_screen_time_hours": 5.34,
            "social_media_hours": 3.23,
            "gaming_hours": 2.05,
            "work_study_hours": 4.66,
            "sleep_hours": 6.54,
            "notifications_per_day": 125.6,
            "app_opens_per_day": 90.0,
            "weekend_screen_time": 7.01,
        },
    },
    1: {
        "name": "Gaming-Heavy High-Exposure",
        "badge": "🔴 Intensive Gaming & Screen Load",
        "addiction_rate": 95.78,
        "cohort_count": 1942,
        "color": "#EF4444",
        "description": (
            "Marked by elevated daily screen exposure (~9.7h), heavy gaming sessions (~3.0h/day), "
            "and extended weekend screen spikes (~11.5h). Highest addiction prevalence in the cohort."
        ),
        "averages": {
            "daily_screen_time_hours": 9.69,
            "social_media_hours": 3.24,
            "gaming_hours": 3.03,
            "work_study_hours": 3.17,
            "sleep_hours": 6.66,
            "notifications_per_day": 135.1,
            "app_opens_per_day": 96.5,
            "weekend_screen_time": 11.47,
        },
    },
    2: {
        "name": "Moderate Screen & Notification-Driven",
        "badge": "🟡 Interrupt-Driven Moderate",
        "addiction_rate": 44.68,
        "cohort_count": 1831,
        "color": "#F59E0B",
        "description": (
            "Lower total screen hours (~5.2h) and healthier sleep (~6.9h), but high notification "
            "frequency (~144/day) and frequent context-switching."
        ),
        "averages": {
            "daily_screen_time_hours": 5.19,
            "social_media_hours": 3.23,
            "gaming_hours": 1.98,
            "work_study_hours": 1.80,
            "sleep_hours": 6.90,
            "notifications_per_day": 143.8,
            "app_opens_per_day": 101.6,
            "weekend_screen_time": 6.87,
        },
    },
    3: {
        "name": "Screen & Social-Heavy",
        "badge": "🟠 High Social & Weekend Binge",
        "addiction_rate": 95.38,
        "cohort_count": 1840,
        "color": "#F97316",
        "description": (
            "Driven by prolonged daily screen time (~9.7h), intensive social media scrolling (~3.4h+), "
            "frequent app opens (~103/day), and heavy weekend usage (~11.5h)."
        ),
        "averages": {
            "daily_screen_time_hours": 9.71,
            "social_media_hours": 3.40,
            "gaming_hours": 0.94,
            "work_study_hours": 3.30,
            "sleep_hours": 6.87,
            "notifications_per_day": 132.8,
            "app_opens_per_day": 103.5,
            "weekend_screen_time": 11.54,
        },
    },
}

RF_FEATURE_IMPORTANCE = pd.DataFrame({
    "Feature": [
        "Social Media Hours",
        "Daily Screen Time",
        "Weekend Screen Time",
        "Sleep Hours",
        "Work / Study Hours",
        "Gaming Hours",
        "Notifications / Day",
        "App Opens / Day",
        "Age",
        "Gender",
    ],
    "Importance": [0.3639, 0.3052, 0.1911, 0.0245, 0.0237, 0.0235, 0.0222, 0.0219, 0.0161, 0.0080],
})

PERSONA_PRESETS: Dict[str, Dict[str, Any]] = {
    "Balanced Professional": {
        "age": 27,
        "gender": "Female",
        "daily_screen_time_hours": 4.5,
        "social_media_hours": 1.5,
        "gaming_hours": 0.5,
        "work_study_hours": 6.0,
        "sleep_hours": 7.5,
        "notifications_per_day": 55,
        "app_opens_per_day": 40,
        "stress_level_numeric": 1,
        "weekend_screen_time": 5.0,
    },
    "Heavy Social Scroller": {
        "age": 21,
        "gender": "Female",
        "daily_screen_time_hours": 9.5,
        "social_media_hours": 5.5,
        "gaming_hours": 1.0,
        "work_study_hours": 3.5,
        "sleep_hours": 5.5,
        "notifications_per_day": 190,
        "app_opens_per_day": 135,
        "stress_level_numeric": 3,
        "weekend_screen_time": 11.5,
    },
    "Competitive Gamer": {
        "age": 19,
        "gender": "Male",
        "daily_screen_time_hours": 10.0,
        "social_media_hours": 2.0,
        "gaming_hours": 5.5,
        "work_study_hours": 2.5,
        "sleep_hours": 5.5,
        "notifications_per_day": 140,
        "app_opens_per_day": 95,
        "stress_level_numeric": 2,
        "weekend_screen_time": 12.5,
    },
    "Moderate Notification-Busy": {
        "age": 29,
        "gender": "Male",
        "daily_screen_time_hours": 5.0,
        "social_media_hours": 2.5,
        "gaming_hours": 1.5,
        "work_study_hours": 1.8,
        "sleep_hours": 7.0,
        "notifications_per_day": 155,
        "app_opens_per_day": 105,
        "stress_level_numeric": 2,
        "weekend_screen_time": 6.5,
    },
}
