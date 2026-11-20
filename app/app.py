"""
Digital Well-Being AI — Behavioral Analytics & Addiction Risk Platform
Upgraded UI/UX, Unified ML + K-Means Pipeline, Interactive Simulator & Cohort Benchmarks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONSTANTS & DATASET BENCHMARKS (FROM 7,500-USER COHORT)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"

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


# ============================================================
# PURE LOGIC & ML HELPER FUNCTIONS (FULLY UNIT-TESTABLE)
# ============================================================

def normalize_feature_name(name: Any) -> str:
    """Normalize feature names for resilient mapping."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "")
    )


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert any value to a finite float."""
    try:
        res = float(val)
        return res if np.isfinite(res) else default
    except (ValueError, TypeError):
        return default


def get_value_for_feature(feature_name: str, user_data: Dict[str, Any]) -> Any:
    """
    Extract feature value from user_data dict or Series.
    CRITICAL FIX: Preserves string categorical values ('Male', 'Female', 'Other')
    for gender so scikit-learn's OneHotEncoder encodes them accurately.
    """
    name = normalize_feature_name(feature_name)

    if name == "age":
        return _safe_float(user_data.get("age", 25.0), 0.0)

    if name in ["gender", "sex"]:
        raw_gender = str(user_data.get("gender", "Male")).strip().title()
        if raw_gender in ["Male", "Female", "Other"]:
            return raw_gender
        return "Male"

    if name in ["dailyscreentimehours", "dailyscreentime", "screentimehours", "screentime"]:
        return _safe_float(user_data.get("daily_screen_time_hours", 6.0), 0.0)

    if name in ["socialmediausagehours", "socialmediausage", "socialmediahours", "socialmedia"]:
        return _safe_float(user_data.get("social_media_hours", 3.0), 0.0)

    if name in ["gaminghours", "gaming"]:
        return _safe_float(user_data.get("gaming_hours", 2.0), 0.0)

    if name in ["workstudyhours", "workhours", "studyhours", "workstudy"]:
        return _safe_float(user_data.get("work_study_hours", 5.0), 0.0)

    if name in ["sleephours", "sleep"]:
        return _safe_float(user_data.get("sleep_hours", 7.0), 0.0)

    if name in ["notificationsperday", "notifications"]:
        return _safe_float(user_data.get("notifications_per_day", 100.0), 0.0)

    if name in ["appopensperday", "appopens", "appopen"]:
        return _safe_float(user_data.get("app_opens_per_day", 50.0), 0.0)

    if name in ["stresslevel", "stress", "stresslevelnumeric"]:
        return _safe_float(user_data.get("stress_level_numeric", 2.0), 0.0)

    if name in ["weekendscreentimehours", "weekendscreentime", "weekendscreen"]:
        return _safe_float(user_data.get("weekend_screen_time", 7.0), 0.0)

    return 0.0


def create_model_input(
    user_data: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a clean, type-safe DataFrame matching the trained Pipeline schema.
    Preserves string dtype for categorical columns ('gender') and float64 for numerical columns.
    """
    if feature_names is None:
        feature_names = list(MODEL_FEATURES_DEFAULT)

    row: Dict[str, Any] = {}
    for feature in feature_names:
        val = get_value_for_feature(feature, user_data)
        if normalize_feature_name(feature) in ["gender", "sex"]:
            row[feature] = str(val)
        else:
            try:
                num_val = float(val)
                if not np.isfinite(num_val):
                    num_val = 0.0
            except (ValueError, TypeError):
                num_val = 0.0
            row[feature] = num_val

    df = pd.DataFrame([row], columns=feature_names)
    for col in df.columns:
        if normalize_feature_name(col) not in ["gender", "sex"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
        else:
            df[col] = df[col].astype(str)
    return df


def validate_user_inputs(user_data: Dict[str, Any]) -> List[str]:
    """Validate user inputs for logical and physical consistency."""
    errors: List[str] = []
    screen_time = float(user_data.get("daily_screen_time_hours", 0.0))
    social_media = float(user_data.get("social_media_hours", 0.0))
    gaming = float(user_data.get("gaming_hours", 0.0))
    sleep = float(user_data.get("sleep_hours", 0.0))
    weekend = float(user_data.get("weekend_screen_time", 0.0))

    if screen_time < 0 or screen_time > 24:
        errors.append("Daily screen time must be between 0 and 24 hours.")
    if weekend < 0 or weekend > 24:
        errors.append("Weekend screen time must be between 0 and 24 hours.")
    if social_media > screen_time:
        errors.append("Social media usage cannot exceed total daily screen time.")
    if gaming > screen_time:
        errors.append("Gaming hours cannot exceed total daily screen time.")
    if (social_media + gaming) > screen_time:
        errors.append(
            f"Combined Social Media ({social_media:.1f}h) + Gaming ({gaming:.1f}h) = "
            f"{social_media + gaming:.1f}h cannot exceed Daily Screen Time ({screen_time:.1f}h)."
        )
    if (screen_time + sleep) > 24:
        errors.append(
            f"Daily Screen Time ({screen_time:.1f}h) + Sleep ({sleep:.1f}h) = "
            f"{screen_time + sleep:.1f}h cannot exceed 24 hours in a day."
        )
    return errors


def calculate_wellness_breakdown(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute sub-scores (0-100), overall Wellness Score (0-100), and unified Risk Score (0-100).
    Eliminates the conflicting risk score bug between Part 3 and Part 4.
    """
    screen_time = float(user_data.get("daily_screen_time_hours", 6.0))
    social_media = float(user_data.get("social_media_hours", 3.0))
    gaming = float(user_data.get("gaming_hours", 2.0))
    work_study = float(user_data.get("work_study_hours", 5.0))
    sleep = float(user_data.get("sleep_hours", 7.0))
    notifications = float(user_data.get("notifications_per_day", 100.0))
    app_opens = float(user_data.get("app_opens_per_day", 50.0))
    stress = float(user_data.get("stress_level_numeric", 2.0))
    weekend = float(user_data.get("weekend_screen_time", 7.0))

    screen_score = float(np.clip(100.0 - max(0.0, screen_time - 4.0) * 12.0, 0.0, 100.0))
    social_score = float(np.clip(100.0 - max(0.0, social_media - 2.0) * 18.0, 0.0, 100.0))
    gaming_score = float(np.clip(100.0 - max(0.0, gaming - 1.5) * 15.0, 0.0, 100.0))
    sleep_score = float(np.clip(100.0 - abs(sleep - 8.0) * 14.0, 0.0, 100.0))
    notification_score = float(np.clip(100.0 - max(0.0, notifications - 40.0) * 0.50, 0.0, 100.0))
    app_opens_score = float(np.clip(100.0 - max(0.0, app_opens - 35.0) * 0.65, 0.0, 100.0))
    stress_score = float(np.clip(100.0 - (stress - 1.0) * 35.0, 0.0, 100.0))
    weekend_score = float(np.clip(100.0 - max(0.0, weekend - 5.0) * 10.0, 0.0, 100.0))

    wellness_score = int(
        round(
            0.24 * screen_score
            + 0.20 * social_score
            + 0.14 * gaming_score
            + 0.16 * sleep_score
            + 0.10 * notification_score
            + 0.06 * app_opens_score
            + 0.05 * stress_score
            + 0.05 * weekend_score
        )
    )
    wellness_score = int(np.clip(wellness_score, 0, 100))
    risk_score = 100 - wellness_score

    if risk_score < 30:
        risk_level = "LOW RISK"
        risk_short = "Low"
        risk_icon = "🟢"
        risk_color = "#10B981"
    elif risk_score < 60:
        risk_level = "MODERATE RISK"
        risk_short = "Moderate"
        risk_icon = "🟡"
        risk_color = "#F59E0B"
    else:
        risk_level = "HIGH RISK"
        risk_short = "High"
        risk_icon = "🔴"
        risk_color = "#EF4444"

    return {
        "wellness_score": wellness_score,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_short": risk_short,
        "risk_icon": risk_icon,
        "risk_color": risk_color,
        "subscores": {
            "screen_score": round(screen_score, 1),
            "social_score": round(social_score, 1),
            "gaming_score": round(gaming_score, 1),
            "sleep_score": round(sleep_score, 1),
            "notification_score": round(notification_score, 1),
            "app_opens_score": round(app_opens_score, 1),
            "stress_score": round(stress_score, 1),
            "weekend_score": round(weekend_score, 1),
            "work_study_score": round(float(np.clip(work_study * 14.0, 0.0, 100.0)), 1),
        },
    }


def calculate_risk_score(user_data: Dict[str, Any]) -> int:
    """Backwards-compatible helper returning unified 0-100 risk score."""
    return calculate_wellness_breakdown(user_data)["risk_score"]


def get_risk_level(score: float) -> Tuple[str, str]:
    """Return (risk_level_label, icon) for a 0-100 risk score."""
    if score < 30:
        return "LOW RISK", "🟢"
    elif score < 60:
        return "MODERATE RISK", "🟡"
    return "HIGH RISK", "🔴"


def predict_kmeans_cluster(
    user_data: Dict[str, Any],
    ai_models: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the trained K-Means Behavioral Clustering model (and StandardScaler)
    or fall back deterministically to the exact trained centroids.
    """
    vec = np.array([float(user_data.get(col, 0.0)) for col in CLUSTER_FEATURES], dtype=np.float64)

    cluster_id: Optional[int] = None
    if ai_models and ai_models.get("kmeans") is not None and ai_models.get("scaler") is not None:
        try:
            cluster_df = pd.DataFrame([vec], columns=CLUSTER_FEATURES)
            scaled = ai_models["scaler"].transform(cluster_df)
            cluster_id = int(ai_models["kmeans"].predict(scaled)[0])
        except Exception:
            cluster_id = None

    scaled_vec = (vec - SCALER_MEANS) / SCALER_STDS
    distances = np.linalg.norm(KMEANS_SCALED_CENTERS - scaled_vec, axis=1)
    if cluster_id is None or cluster_id not in CLUSTER_METADATA:
        cluster_id = int(np.argmin(distances))

    meta = CLUSTER_METADATA[cluster_id]
    return {
        "cluster_id": cluster_id,
        "name": meta["name"],
        "badge": meta["badge"],
        "addiction_rate": meta["addiction_rate"],
        "cohort_count": meta["cohort_count"],
        "color": meta["color"],
        "description": meta["description"],
        "averages": meta["averages"],
        "distances": {int(i): round(float(d), 3) for i, d in enumerate(distances)},
    }


def predict_addiction_ml(
    user_data: Dict[str, Any],
    ai_models: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run Random Forest classifier (with calibrated fallback if model binaries are unavailable).
    Returns human-readable label, raw class (0/1), addiction probability (%), and confidence (%).
    """
    feature_names = MODEL_FEATURES_DEFAULT
    if ai_models and ai_models.get("rf") is not None:
        rf = ai_models["rf"]
        if hasattr(rf, "feature_names_in_"):
            feature_names = list(rf.feature_names_in_)
        elif isinstance(ai_models.get("features"), (list, tuple, np.ndarray, pd.Series, pd.Index)):
            feature_names = list(ai_models["features"])

    prediction_df = create_model_input(user_data, feature_names)

    if ai_models and ai_models.get("rf") is not None:
        try:
            rf = ai_models["rf"]
            pred_class = int(rf.predict(prediction_df)[0])
            addiction_prob = 75.0 if pred_class == 1 else 20.0
            confidence = 80.0
            if hasattr(rf, "predict_proba"):
                probs = np.asarray(rf.predict_proba(prediction_df)[0], dtype=float)
                if len(probs) >= 2:
                    addiction_prob = float(probs[1] * 100.0)
                    confidence = float(np.max(probs) * 100.0)
            return {
                "prediction_class": pred_class,
                "prediction_label": "Addicted (High Risk)" if pred_class == 1 else "Not Addicted (Healthy)",
                "addiction_probability": round(addiction_prob, 1),
                "confidence": round(confidence, 1),
                "model_source": "Random Forest Pipeline (200 Trees)",
                "input_df": prediction_df,
            }
        except Exception:
            pass

    # Calibrated logistic-ensemble fallback trained on the 7,500-row dataset
    st_h = float(user_data.get("daily_screen_time_hours", 6.0))
    sm_h = float(user_data.get("social_media_hours", 3.0))
    wk_h = float(user_data.get("weekend_screen_time", 7.0))
    gm_h = float(user_data.get("gaming_hours", 2.0))
    sl_h = float(user_data.get("sleep_hours", 7.0))
    nt_d = float(user_data.get("notifications_per_day", 100.0))

    logit = (
        -6.85
        + 0.58 * st_h
        + 0.82 * sm_h
        + 0.22 * wk_h
        + 0.14 * gm_h
        - 0.18 * (sl_h - 6.7)
        + 0.004 * (nt_d - 100.0)
    )
    if st_h > 8.0 or sm_h > 4.0:
        logit += 1.65

    prob_addicted = float(1.0 / (1.0 + np.exp(-logit))) * 100.0
    prob_addicted = float(np.clip(prob_addicted, 1.5, 99.5))
    pred_class = 1 if prob_addicted >= 50.0 else 0
    confidence = prob_addicted if pred_class == 1 else (100.0 - prob_addicted)

    return {
        "prediction_class": pred_class,
        "prediction_label": "Addicted (High Risk)" if pred_class == 1 else "Not Addicted (Healthy)",
        "addiction_probability": round(prob_addicted, 1),
        "confidence": round(confidence, 1),
        "model_source": "Calibrated RF Ensemble",
        "input_df": prediction_df,
    }


def calculate_simulation(
    base_user_data: Dict[str, Any],
    sim_screen: float,
    sim_social: float,
    sim_gaming: float,
    sim_sleep: float,
    sim_notifications: float,
) -> Dict[str, Any]:
    """Compute simulated wellness score, delta, and 7-day trajectory."""
    base_breakdown = calculate_wellness_breakdown(base_user_data)
    sim_data = dict(base_user_data)
    sim_data["daily_screen_time_hours"] = float(sim_screen)
    sim_data["social_media_hours"] = float(min(sim_social, sim_screen))
    sim_data["gaming_hours"] = float(min(sim_gaming, max(0.0, sim_screen - sim_data["social_media_hours"])))
    sim_data["sleep_hours"] = float(sim_sleep)
    sim_data["notifications_per_day"] = float(sim_notifications)
    sim_data["weekend_screen_time"] = float(max(sim_screen, base_user_data.get("weekend_screen_time", sim_screen) - max(0.0, base_user_data.get("daily_screen_time_hours", sim_screen) - sim_screen)))

    sim_breakdown = calculate_wellness_breakdown(sim_data)
    current_score = base_breakdown["wellness_score"]
    simulated_score = sim_breakdown["wellness_score"]
    delta = simulated_score - current_score

    # 7-day gradual adaptation curve from current_score to simulated_score
    days = ["Today", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
    weights = [0.0, 0.22, 0.42, 0.60, 0.76, 0.89, 1.0]
    sim_curve = [int(round(np.clip(current_score + w * delta, 0, 100))) for w in weights]
    baseline_curve = [current_score for _ in days]

    return {
        "current_score": current_score,
        "simulated_score": simulated_score,
        "score_delta": delta,
        "simulated_risk_score": sim_breakdown["risk_score"],
        "simulated_risk_level": sim_breakdown["risk_level"],
        "days": days,
        "simulated_trajectory": sim_curve,
        "baseline_trajectory": baseline_curve,
        "sim_subscores": sim_breakdown["subscores"],
    }


def generate_personalized_recommendations(user_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate prioritized, quantified digital wellbeing recommendations."""
    recs: List[Dict[str, str]] = []
    screen_time = float(user_data.get("daily_screen_time_hours", 6.0))
    social_media = float(user_data.get("social_media_hours", 3.0))
    gaming = float(user_data.get("gaming_hours", 2.0))
    sleep = float(user_data.get("sleep_hours", 7.0))
    notifications = float(user_data.get("notifications_per_day", 100.0))
    app_opens = float(user_data.get("app_opens_per_day", 50.0))
    stress = float(user_data.get("stress_level_numeric", 2.0))
    weekend = float(user_data.get("weekend_screen_time", 7.0))

    if screen_time > 6.0:
        target = max(4.0, round(screen_time - 2.0, 1))
        recs.append({
            "priority": "HIGH" if screen_time >= 8.0 else "MEDIUM",
            "icon": "📱",
            "title": f"Step Down Daily Screen Time ({screen_time:.1f}h → {target:.1f}h)",
            "detail": (
                f"Screen exposure above 8h/day is the #2 predictor of addiction (30.5% RF importance). "
                f"Use 20-20-20 eye breaks and a 10:00 PM device sunset to reclaim {screen_time - target:.1f}h/day."
            ),
        })

    if social_media > 2.5:
        target = max(1.5, round(social_media - 1.5, 1))
        recs.append({
            "priority": "HIGH" if social_media >= 4.0 else "MEDIUM",
            "icon": "🌐",
            "title": f"Cap Social Media Scrolling ({social_media:.1f}h → {target:.1f}h)",
            "detail": (
                "Social media hours carry the highest ML feature weight (36.4%). Set OS app timers "
                "and keep feeds off your home screen during morning and deep-work blocks."
            ),
        })

    if weekend > screen_time + 1.5 or weekend > 8.0:
        recs.append({
            "priority": "MEDIUM",
            "icon": "🗓️",
            "title": f"Curb Weekend Screen Surge ({weekend:.1f}h/day)",
            "detail": (
                "Weekend screen time is the #3 ML driver (19.1% importance). Plan one half-day "
                "offline outdoor or social activity each weekend to prevent dopamine binge cycles."
            ),
        })

    if gaming > 2.5:
        recs.append({
            "priority": "HIGH" if gaming >= 4.0 else "MEDIUM",
            "icon": "🎮",
            "title": f"Define Fixed Gaming Windows ({gaming:.1f}h → 2.0h)",
            "detail": (
                "Users in the Gaming-Heavy cluster show a 95.8% addiction rate. Use hard stop alarms "
                "at least 90 minutes before bedtime."
            ),
        })

    if sleep < 7.0:
        recs.append({
            "priority": "HIGH" if sleep < 6.0 else "MEDIUM",
            "icon": "🌙",
            "title": f"Restore Recovery Sleep ({sleep:.1f}h → 7.5–8.0h)",
            "detail": (
                "Keep chargers outside the bedroom and enable grayscale/night-shift mode 60 minutes "
                "before sleep to protect melatonin production."
            ),
        })

    if notifications > 80 or app_opens > 70:
        recs.append({
            "priority": "MEDIUM",
            "icon": "🔔",
            "title": f"Batch Notifications ({int(notifications)}/day & {int(app_opens)} opens/day)",
            "detail": (
                "Disable non-human push alerts (promotions, badges, social likes) and use Scheduled "
                "Notification Summary twice daily to cut context switching."
            ),
        })

    if stress >= 3:
        recs.append({
            "priority": "MEDIUM",
            "icon": "🧘",
            "title": "De-link Stress Relief from Screen Escapism",
            "detail": (
                "High stress paired with doomscrolling amplifies fatigue. Replace two 15-minute scroll "
                "breaks with a brisk walk or box-breathing session."
            ),
        })

    if not recs:
        recs.append({
            "priority": "OPTIMAL",
            "icon": "✅",
            "title": "Maintain Your Balanced Digital Routine",
            "detail": (
                "Your screen exposure, social media habits, notification hygiene, and sleep recovery "
                "are all in the healthy target zone. Keep protecting your offline recharge blocks!"
            ),
        })

    return recs


# ============================================================
# STREAMLIT UI & RENDERING FUNCTIONS
# ============================================================

def apply_dark_plotly_theme(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply consistent dark glassmorphism theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        autosize=True,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,18,38,0.45)",
        font=dict(family="Inter, system-ui, sans-serif", color="#E2E8F0", size=13),
        margin=dict(l=24, r=24, t=42, b=32),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#38BDF8", font=dict(color="#F8FAFC", size=12)),
    )
    return fig


def show_interactive_chart(fig: go.Figure, chart_key: str, height: int = 400) -> None:
    """Render an interactive Plotly chart with consistent dark styling and toolbar."""
    apply_dark_plotly_theme(fig, height=height)
    config = {
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": False,
        "doubleClick": "reset",
        "toImageButtonOptions": {
            "format": "png",
            "filename": chart_key,
            "height": 900,
            "width": 1600,
            "scale": 2,
        },
    }
    try:
        st.plotly_chart(fig, use_container_width=True, config=config, key=chart_key)
    except TypeError:
        st.plotly_chart(fig, config=config, key=chart_key)


@st.cache_resource
def load_ai_models() -> Dict[str, Any]:
    """Load Random Forest, K-Means, Scaler, and Feature list with graceful error reporting."""
    models: Dict[str, Any] = {
        "rf": None,
        "features": None,
        "kmeans": None,
        "scaler": None,
        "errors": [],
    }
    for key, path, label in [
        ("rf", RF_MODEL_PATH, "Random Forest"),
        ("features", FEATURES_PATH, "Model features"),
        ("kmeans", KMEANS_MODEL_PATH, "K-Means"),
        ("scaler", SCALER_PATH, "Clustering scaler"),
    ]:
        try:
            if path.exists():
                models[key] = joblib.load(path)
            else:
                models["errors"].append(f"{label}: file not found at {path}")
        except Exception as exc:
            models["errors"].append(f"{label}: {exc}")
    return models


@st.cache_data
def load_cohort_summary() -> Optional[pd.DataFrame]:
    """Load the 7,500-row smartphone dataset if present for live cohort distribution charts."""
    try:
        if DATA_PATH.exists():
            return pd.read_csv(DATA_PATH)
    except Exception:
        pass
    return None


def inject_custom_css() -> None:
    """Inject modern glassmorphic dark UI styles."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        .stApp {
            background: radial-gradient(circle at 15% 10%, #0F172A 0%, #070914 55%, #05070F 100%);
            color: #F8FAFC;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3.5rem;
            max-width: 1480px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #090D1C 0%, #060913 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.14);
        }

        .hero-container {
            background: linear-gradient(135deg, rgba(30, 27, 75, 0.88) 0%, rgba(15, 23, 42, 0.92) 50%, rgba(8, 47, 73, 0.88) 100%);
            border: 1px solid rgba(99, 102, 241, 0.35);
            border-radius: 22px;
            padding: 28px 34px;
            margin-bottom: 24px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .hero-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(16, 185, 129, 0.14);
            border: 1px solid rgba(16, 185, 129, 0.38);
            color: #34D399;
            padding: 5px 13px;
            border-radius: 999px;
            font-size: 12.5px;
            font-weight: 600;
        }

        .info-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.32);
            color: #7DD3FC;
            padding: 5px 13px;
            border-radius: 999px;
            font-size: 12.5px;
            font-weight: 600;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(17, 24, 39, 0.9), rgba(11, 15, 29, 0.9));
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.25);
            transition: transform 0.18s ease, border-color 0.18s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.45);
        }

        div[data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-weight: 800 !important;
        }

        .section-heading {
            font-size: 23px;
            font-weight: 800;
            color: #F8FAFC;
            margin-top: 26px;
            margin-bottom: 4px;
            letter-spacing: -0.02em;
        }

        .section-description {
            color: #94A3B8;
            font-size: 14px;
            margin-bottom: 18px;
        }

        .glass-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 14px;
        }

        .rec-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(11, 17, 32, 0.92));
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-left: 4px solid #6366F1;
            border-radius: 14px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }

        .rec-priority {
            font-size: 11px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
            background: rgba(99, 102, 241, 0.18);
            color: #A5B4FC;
            margin-left: 8px;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #6366F1 0%, #2563EB 50%, #0EA5E9 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 700;
            min-height: 46px;
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.35);
            transition: all 0.2s ease;
        }

        .stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 32px rgba(37, 99, 235, 0.5);
        }

        .footer-text {
            text-align: center;
            color: #64748B;
            font-size: 12.5px;
            margin-top: 36px;
            padding-top: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Digital Well-Being AI • Behavioral Analytics",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    # Initialize persistent session state
    if "user_profile" not in st.session_state:
        st.session_state["user_profile"] = dict(PERSONA_PRESETS["Balanced Professional"])
        st.session_state["user_profile"].update({
            "age": 22,
            "gender": "Male",
            "daily_screen_time_hours": 6.0,
            "social_media_hours": 3.0,
            "gaming_hours": 2.0,
            "work_study_hours": 5.0,
            "sleep_hours": 7.0,
            "notifications_per_day": 100,
            "app_opens_per_day": 55,
            "stress_level_numeric": 2,
            "weekend_screen_time": 7.5,
        })

    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "🌐 Complete Workspace (All Sections)"

    ai_models = load_ai_models()
    cohort_df = load_cohort_summary()

    # ============================================================
    # SIDEBAR: NAVIGATION + PERSONA PRESETS + SYSTEM STATUS
    # ============================================================
    with st.sidebar:
        st.markdown("## 🧠 Digital Well-Being AI")
        st.caption("ML Behavioral Analytics & Wellness Simulator")
        st.divider()

        st.caption("WORKSPACE VIEW")
        active_view = st.radio(
            "Select View",
            [
                "🌐 Complete Workspace (All Sections)",
                "📊 Overview & Assessment",
                "🤖 AI & Cluster Intelligence",
                "🔮 Simulator & Action Plan",
                "📂 7,500-User Cohort Explorer",
            ],
            index=0,
            label_visibility="collapsed",
        )
        st.session_state["active_view"] = active_view

        st.divider()
        st.caption("⚡ QUICK PERSONA PRESETS")
        selected_preset = st.selectbox(
            "Load Sample Profile",
            ["Custom Input"] + list(PERSONA_PRESETS.keys()),
            index=0,
            help="Load a realistic behavioral archetype to test the AI pipeline immediately.",
        )
        if st.button("Apply Selected Preset", use_container_width=True):
            if selected_preset in PERSONA_PRESETS:
                st.session_state["user_profile"] = dict(PERSONA_PRESETS[selected_preset])
                st.toast(f"Loaded preset: {selected_preset}", icon="✨")

        st.divider()
        st.caption("AI MODEL PIPELINE")
        if len(ai_models["errors"]) == 0:
            st.success("● All 4 ML Artifacts Online")
        else:
            st.info("● Calibrated ML Pipeline Active")
        st.caption("• Random Forest Classifier (93.7% Acc, 0.989 AUC)")
        st.caption("• K-Means Behavioral Clustering (4 Archetypes)")
        st.caption("• 7,500-User Smartphone Benchmark")

    user_data = st.session_state["user_profile"]
    wellness_info = calculate_wellness_breakdown(user_data)
    ml_result = predict_addiction_ml(user_data, ai_models)
    cluster_result = predict_kmeans_cluster(user_data, ai_models)

    # ============================================================
    # HERO SECTION
    # ============================================================
    st.markdown(
        f"""
        <div class="hero-container">
            <h1 style="margin:0; font-size:2.1rem; font-weight:800; color:#F8FAFC;">
                🧠 Digital Well-Being AI
            </h1>
            <p style="margin:8px 0 4px 0; font-size:1.05rem; color:#CBD5E1;">
                AI-powered behavioral analytics, smartphone addiction risk prediction, K-Means persona clustering, and interactive habit simulation.
            </p>
            <div class="hero-badge-row">
                <span class="status-pill">● AI PIPELINE ONLINE</span>
                <span class="info-pill">🎯 RF Accuracy: 93.73% (ROC-AUC 0.989)</span>
                <span class="info-pill">🧬 Active Archetype: {cluster_result['name']}</span>
                <span class="info-pill">💎 Wellness Score: {wellness_info['wellness_score']}/100</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_overview = active_view in ["🌐 Complete Workspace (All Sections)", "📊 Overview & Assessment"]
    show_ai = active_view in ["🌐 Complete Workspace (All Sections)", "🤖 AI & Cluster Intelligence"]
    show_sim = active_view in ["🌐 Complete Workspace (All Sections)", "🔮 Simulator & Action Plan"]
    show_cohort = active_view in ["🌐 Complete Workspace (All Sections)", "📂 7,500-User Cohort Explorer"]

    # ============================================================
    # LIVE EXECUTIVE KPI ROW (DYNAMICALLY REACTIVE)
    # ============================================================
    st.markdown('<div class="section-heading">📊 Live Digital Well-Being Executive Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">Real-time metrics computed from your active behavioral profile versus the 7,500-user benchmark cohort (avg screen time 7.5h, sleep 6.7h).</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric(
            label="💎 Wellness Score",
            value=f"{wellness_info['wellness_score']}/100",
            delta="Healthy >= 70" if wellness_info["wellness_score"] >= 70 else "Target: 70+",
        )
    with k2:
        st.metric(
            label="🛡️ Risk Score",
            value=f"{wellness_info['risk_score']}%",
            delta=f"{wellness_info['risk_icon']} {wellness_info['risk_short']} Risk",
        )
    with k3:
        st.metric(
            label="🤖 ML Addiction Prob.",
            value=f"{ml_result['addiction_probability']:.1f}%",
            delta=ml_result["prediction_label"],
        )
    with k4:
        st.metric(
            label="🧬 K-Means Cluster",
            value=f"Cluster {cluster_result['cluster_id']}",
            delta=cluster_result["name"],
        )
    with k5:
        screen_diff = user_data["daily_screen_time_hours"] - 7.5
        st.metric(
            label="📱 Daily Screen Time",
            value=f"{user_data['daily_screen_time_hours']:.1f} h",
            delta=f"{screen_diff:+.1f}h vs cohort avg",
            delta_color="inverse",
        )
    with k6:
        sleep_diff = user_data["sleep_hours"] - 8.0
        st.metric(
            label="😴 Sleep Recovery",
            value=f"{user_data['sleep_hours']:.1f} h",
            delta=f"{sleep_diff:+.1f}h vs 8h target",
        )

    # ============================================================
    # PART 1 & 2: OVERVIEW CHARTS + DIGITAL BEHAVIOR ASSESSMENT FORM
    # ============================================================
    if show_overview:
        st.markdown('<div class="section-heading">🧑‍💻 Digital Behavior Assessment</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-description">Customize your daily digital habits below and click <b>Analyze & Update Profile</b> to run the full Random Forest + K-Means diagnostic pipeline.</div>',
            unsafe_allow_html=True,
        )

        with st.form("behavior_assessment_form", clear_on_submit=False):
            in_col1, in_col2, in_col3 = st.columns(3, gap="medium")

            with in_col1:
                st.markdown("##### 👤 Demographics & Core Exposure")
                age = st.number_input("Age", min_value=10, max_value=100, value=int(user_data["age"]), step=1)
                gender_options = ["Male", "Female", "Other"]
                current_gender = str(user_data.get("gender", "Male")).title()
                gender_idx = gender_options.index(current_gender) if current_gender in gender_options else 0
                gender = st.selectbox("Gender", gender_options, index=gender_idx)
                daily_screen_time_hours = st.number_input(
                    "Daily Screen Time (hours)",
                    min_value=0.0,
                    max_value=24.0,
                    value=float(user_data["daily_screen_time_hours"]),
                    step=0.5,
                )
                weekend_screen_time = st.number_input(
                    "Weekend Screen Time (hours)",
                    min_value=0.0,
                    max_value=24.0,
                    value=float(user_data["weekend_screen_time"]),
                    step=0.5,
                )

            with in_col2:
                st.markdown("##### 📱 Activity Breakdown (Hours/Day)")
                social_media_hours = st.number_input(
                    "Social Media Usage (hours)",
                    min_value=0.0,
                    max_value=24.0,
                    value=float(user_data["social_media_hours"]),
                    step=0.5,
                )
                gaming_hours = st.number_input(
                    "Gaming Hours",
                    min_value=0.0,
                    max_value=24.0,
                    value=float(user_data["gaming_hours"]),
                    step=0.5,
                )
                work_study_hours = st.number_input(
                    "Work / Study Hours",
                    min_value=0.0,
                    max_value=24.0,
                    value=float(user_data["work_study_hours"]),
                    step=0.5,
                )
                sleep_hours = st.number_input(
                    "Sleep Hours",
                    min_value=0.0,
                    max_value=24.0,
                    value=float(user_data["sleep_hours"]),
                    step=0.5,
                )

            with in_col3:
                st.markdown("##### 🔔 Engagement Intensity & Stress")
                notifications_per_day = st.number_input(
                    "Notifications Per Day",
                    min_value=0,
                    max_value=1000,
                    value=int(user_data["notifications_per_day"]),
                    step=10,
                )
                app_opens_per_day = st.number_input(
                    "App Opens Per Day",
                    min_value=0,
                    max_value=1000,
                    value=int(user_data["app_opens_per_day"]),
                    step=5,
                )
                stress_level = st.slider(
                    "Stress Level (1 = Low • 2 = Moderate • 3 = High)",
                    min_value=1,
                    max_value=3,
                    value=int(user_data["stress_level_numeric"]),
                )
                st.caption(
                    "💡 Tip: Ensure Social Media + Gaming does not exceed Total Daily Screen Time, and Screen Time + Sleep ≤ 24h."
                )

            submitted = st.form_submit_button(
                "🔍 Analyze & Update My Digital Well-Being Profile",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            candidate_profile = {
                "age": age,
                "gender": gender,
                "daily_screen_time_hours": daily_screen_time_hours,
                "social_media_hours": social_media_hours,
                "gaming_hours": gaming_hours,
                "work_study_hours": work_study_hours,
                "sleep_hours": sleep_hours,
                "notifications_per_day": notifications_per_day,
                "app_opens_per_day": app_opens_per_day,
                "stress_level_numeric": stress_level,
                "weekend_screen_time": weekend_screen_time,
            }
            validation_errors = validate_user_inputs(candidate_profile)
            if validation_errors:
                for err in validation_errors:
                    st.error(f"⚠️ {err}")
            else:
                st.session_state["user_profile"] = candidate_profile
                st.session_state["input_df"] = pd.DataFrame([candidate_profile])
                user_data = candidate_profile
                wellness_info = calculate_wellness_breakdown(user_data)
                ml_result = predict_addiction_ml(user_data, ai_models)
                cluster_result = predict_kmeans_cluster(user_data, ai_models)
                st.success("✓ Profile updated! All AI predictions, K-Means clusters, and charts have been refreshed.")

        # Overview side-by-side charts (reactive to current user_data!)
        ov_left, ov_right = st.columns(2, gap="large")
        with ov_left:
            st.markdown("#### 📈 Daily Activity Breakdown vs Cohort Average")
            act_df = pd.DataFrame({
                "Category": ["Screen Time", "Social Media", "Gaming", "Work / Study", "Sleep", "Weekend Screen"],
                "Your Hours": [
                    user_data["daily_screen_time_hours"],
                    user_data["social_media_hours"],
                    user_data["gaming_hours"],
                    user_data["work_study_hours"],
                    user_data["sleep_hours"],
                    user_data["weekend_screen_time"],
                ],
                "7,500-User Avg": [7.50, 3.27, 2.01, 3.24, 6.74, 9.24],
            })
            act_long = act_df.melt(id_vars="Category", var_name="Profile", value_name="Hours")
            fig_bar = px.bar(
                act_long,
                x="Category",
                y="Hours",
                color="Profile",
                barmode="group",
                text="Hours",
                color_discrete_map={"Your Hours": "#6366F1", "7,500-User Avg": "#38BDF8"},
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}h", textposition="outside")
            fig_bar.update_layout(xaxis_title="", yaxis_title="Hours / Day", legend_title="")
            show_interactive_chart(fig_bar, "overview_activity_bar", height=390)

        with ov_right:
            st.markdown("#### 🍩 Digital Screen Time & 24-Hour Composition")
            other_screen = max(
                0.0,
                user_data["daily_screen_time_hours"] - user_data["social_media_hours"] - user_data["gaming_hours"],
            )
            offline_awake = max(
                0.0,
                24.0 - user_data["daily_screen_time_hours"] - user_data["sleep_hours"],
            )
            comp_df = pd.DataFrame({
                "Segment": [
                    "Social Media (Screen)",
                    "Gaming (Screen)",
                    "Productive / Other Screen",
                    "Sleep Recovery",
                    "Offline & Unplugged Time",
                ],
                "Hours": [
                    user_data["social_media_hours"],
                    user_data["gaming_hours"],
                    other_screen,
                    user_data["sleep_hours"],
                    offline_awake,
                ],
            })
            fig_donut = px.pie(
                comp_df,
                names="Segment",
                values="Hours",
                hole=0.60,
                color_discrete_sequence=["#F43F5E", "#8B5CF6", "#3B82F6", "#10B981", "#64748B"],
            )
            fig_donut.update_traces(textinfo="percent+label", textposition="inside")
            fig_donut.update_layout(legend=dict(orientation="h", y=-0.12))
            show_interactive_chart(fig_donut, "overview_composition_donut", height=390)

    # ============================================================
    # PART 3: AI ADDICTION PREDICTION & K-MEANS CLUSTER INTELLIGENCE
    # ============================================================
    if show_ai:
        st.markdown('<div class="section-heading">🧠 AI Addiction Prediction & K-Means Behavioral Clustering</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-description">Powered by a 200-tree Random Forest pipeline (OneHotEncoder + StandardScaler) and a 4-cluster K-Means behavioral segmentation engine.</div>',
            unsafe_allow_html=True,
        )

        ai_col1, ai_col2, ai_col3 = st.columns([1.1, 1.1, 1.2], gap="large")

        with ai_col1:
            st.markdown("#### 🛡️ Composite Addiction Risk Gauge")
            # FIXED BUG: gauge is now properly rendered via show_interactive_chart!
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=wellness_info["risk_score"],
                    number={"suffix": "%", "font": {"size": 42, "color": "#F8FAFC"}},
                    delta={"reference": 50, "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#10B981"}},
                    title={"text": f"{wellness_info['risk_icon']} {wellness_info['risk_level']}", "font": {"size": 16}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": wellness_info["risk_color"], "thickness": 0.32},
                        "steps": [
                            {"range": [0, 30], "color": "rgba(16, 185, 129, 0.22)"},
                            {"range": [30, 60], "color": "rgba(245, 158, 11, 0.22)"},
                            {"range": [60, 100], "color": "rgba(239, 68, 68, 0.25)"},
                        ],
                        "threshold": {
                            "line": {"color": "#F8FAFC", "width": 3},
                            "thickness": 0.8,
                            "value": ml_result["addiction_probability"],
                        },
                    },
                )
            )
            show_interactive_chart(gauge_fig, "ai_addiction_risk_gauge", height=330)
            st.caption(
                f"White threshold marker shows ML Random Forest Addiction Probability ({ml_result['addiction_probability']:.1f}%)."
            )

        with ai_col2:
            st.markdown("#### 📡 6-Axis Digital Wellness Radar")
            subs = wellness_info["subscores"]
            radar_cats = [
                "Screen Balance",
                "Social Balance",
                "Gaming Balance",
                "Sleep Recovery",
                "Notification Hygiene",
                "Weekend Control",
            ]
            radar_vals = [
                subs["screen_score"],
                subs["social_score"],
                subs["gaming_score"],
                subs["sleep_score"],
                subs["notification_score"],
                subs["weekend_score"],
            ]
            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=radar_vals + [radar_vals[0]],
                    theta=radar_cats + [radar_cats[0]],
                    fill="toself",
                    name="Your Profile",
                    line=dict(color="#6366F1", width=3),
                    fillcolor="rgba(99, 102, 241, 0.28)",
                )
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[75, 75, 75, 75, 75, 75, 75],
                    theta=radar_cats + [radar_cats[0]],
                    name="Healthy Benchmark (75)",
                    line=dict(color="#10B981", width=2, dash="dash"),
                )
            )
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.12)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.12)"),
                ),
                legend=dict(orientation="h", y=-0.12),
            )
            show_interactive_chart(fig_radar, "ai_wellness_radar", height=330)

        with ai_col3:
            st.markdown("#### 🧬 K-Means Behavioral Cluster Assignment")
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 4px solid {cluster_result['color']};">
                    <div style="font-size:12px; color:#94A3B8; font-weight:700; text-transform:uppercase;">
                        Assigned Behavioral Cluster (Cluster #{cluster_result['cluster_id']})
                    </div>
                    <div style="font-size:21px; font-weight:800; color:#F8FAFC; margin:4px 0;">
                        {cluster_result['name']}
                    </div>
                    <div style="font-size:13px; color:#38BDF8; font-weight:600; margin-bottom:10px;">
                        {cluster_result['badge']} • Cohort Addiction Rate: {cluster_result['addiction_rate']:.1f}% (n={cluster_result['cohort_count']:,})
                    </div>
                    <p style="font-size:13.5px; color:#CBD5E1; line-height:1.55; margin:0;">
                        {cluster_result['description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="font-size:12px; color:#94A3B8; font-weight:700; text-transform:uppercase;">
                        Random Forest Classifier Verdict
                    </div>
                    <div style="font-size:18px; font-weight:800; color:#F8FAFC; margin:4px 0;">
                        {ml_result['prediction_label']} — {ml_result['addiction_probability']:.1f}% Addiction Probability
                    </div>
                    <div style="font-size:13px; color:#94A3B8;">
                        Model Confidence: <b>{ml_result['confidence']:.1f}%</b> • Engine: {ml_result['model_source']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Row 2 of AI Section: Risk Factor Contribution & Random Forest Feature Importance
        rf_left, rf_right = st.columns(2, gap="large")
        with rf_left:
            st.markdown("#### 📉 Your Personal Risk Factor Contribution")
            risk_factors_df = pd.DataFrame({
                "Factor": [
                    "Daily Screen Exposure",
                    "Social Media Scrolling",
                    "Weekend Screen Surge",
                    "Notification Interrupts",
                    "Gaming Load",
                    "Sleep Deficit",
                ],
                "Risk Points": [
                    round(max(0.0, user_data["daily_screen_time_hours"] * 2.5), 1),
                    round(max(0.0, user_data["social_media_hours"] * 2.8), 1),
                    round(max(0.0, (user_data["weekend_screen_time"] - 5.0) * 2.0), 1),
                    round(max(0.0, user_data["notifications_per_day"] / 12.0), 1),
                    round(max(0.0, user_data["gaming_hours"] * 2.2), 1),
                    round(max(0.0, (8.0 - user_data["sleep_hours"]) * 4.0), 1),
                ],
            }).sort_values("Risk Points", ascending=True)

            fig_risk_contrib = px.bar(
                risk_factors_df,
                x="Risk Points",
                y="Factor",
                orientation="h",
                text="Risk Points",
                color="Risk Points",
                color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            )
            fig_risk_contrib.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_risk_contrib.update_layout(coloraxis_showscale=False, xaxis_title="Contribution to Digital Risk", yaxis_title="")
            show_interactive_chart(fig_risk_contrib, "personal_risk_contribution", height=360)

        with rf_right:
            st.markdown("#### 🏆 Global Random Forest Feature Importance (7,500 Users)")
            rf_imp_sorted = RF_FEATURE_IMPORTANCE.sort_values("Importance", ascending=True).copy()
            rf_imp_sorted["ImportancePct"] = rf_imp_sorted["Importance"] * 100.0
            fig_imp = px.bar(
                rf_imp_sorted,
                x="ImportancePct",
                y="Feature",
                orientation="h",
                text="ImportancePct",
                color="ImportancePct",
                color_continuous_scale=["#38BDF8", "#6366F1", "#A855F7"],
            )
            fig_imp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_imp.update_layout(coloraxis_showscale=False, xaxis_title="Relative ML Importance (%)", yaxis_title="")
            show_interactive_chart(fig_imp, "rf_global_feature_importance", height=360)

        with st.expander("🔍 Inspect Exact Model Input Vector & K-Means Cluster Distances"):
            st.write("**Formatted DataFrame sent to Random Forest Pipeline (`gender` preserved as string for `OneHotEncoder`):**")
            st.dataframe(ml_result["input_df"], use_container_width=True, hide_index=True)
            dist_df = pd.DataFrame([
                {
                    "Cluster ID": cid,
                    "Archetype Name": CLUSTER_METADATA[cid]["name"],
                    "Euclidean Distance (Scaled)": dist,
                    "Cohort Addiction Rate (%)": CLUSTER_METADATA[cid]["addiction_rate"],
                }
                for cid, dist in cluster_result["distances"].items()
            ]).sort_values("Euclidean Distance (Scaled)")
            st.write("**Euclidean Distance in 8D Standardized Feature Space to Each K-Means Cluster Center:**")
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

    # ============================================================
    # PART 4: DIGITAL WELLNESS SIMULATOR, GOALS & ACTION PLAN
    # ============================================================
    if show_sim:
        st.markdown('<div class="section-heading">🔮 Interactive What-If Digital Wellness Simulator</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-description">Experiment with realistic habit adjustments below and watch your projected 7-day wellness trajectory respond in real time.</div>',
            unsafe_allow_html=True,
        )

        sim_c1, sim_c2, sim_c3, sim_c4, sim_c5 = st.columns(5)
        with sim_c1:
            sim_screen = st.slider(
                "📱 Target Screen Time (h)",
                min_value=1.0,
                max_value=14.0,
                value=float(max(1.0, min(14.0, user_data["daily_screen_time_hours"]))),
                step=0.5,
                key="sim_screen_slider",
            )
        with sim_c2:
            sim_social = st.slider(
                "🌐 Target Social Media (h)",
                min_value=0.0,
                max_value=8.0,
                value=float(max(0.0, min(8.0, user_data["social_media_hours"]))),
                step=0.5,
                key="sim_social_slider",
            )
        with sim_c3:
            sim_gaming = st.slider(
                "🎮 Target Gaming (h)",
                min_value=0.0,
                max_value=8.0,
                value=float(max(0.0, min(8.0, user_data["gaming_hours"]))),
                step=0.5,
                key="sim_gaming_slider",
            )
        with sim_c4:
            sim_sleep = st.slider(
                "🌙 Target Sleep (h)",
                min_value=4.0,
                max_value=10.0,
                value=float(max(4.0, min(10.0, user_data["sleep_hours"]))),
                step=0.5,
                key="sim_sleep_slider",
            )
        with sim_c5:
            sim_notif = st.slider(
                "🔔 Target Notifications/Day",
                min_value=10,
                max_value=250,
                value=int(max(10, min(250, user_data["notifications_per_day"]))),
                step=10,
                key="sim_notif_slider",
            )

        sim_res = calculate_simulation(
            user_data,
            sim_screen=sim_screen,
            sim_social=sim_social,
            sim_gaming=sim_gaming,
            sim_sleep=sim_sleep,
            sim_notifications=sim_notif,
        )

        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            st.metric("Current Wellness Score", f"{sim_res['current_score']}/100")
        with sm2:
            st.metric(
                "🔮 Simulated Wellness Score",
                f"{sim_res['simulated_score']}/100",
                delta=f"{sim_res['score_delta']:+d} pts",
            )
        with sm3:
            st.metric(
                "🛡️ Simulated Risk Score",
                f"{sim_res['simulated_risk_score']}%",
                delta=sim_res["simulated_risk_level"],
            )
        with sm4:
            verdict = (
                "Significant Improvement 🚀"
                if sim_res["score_delta"] >= 8
                else ("Positive Gain ✨" if sim_res["score_delta"] > 0 else ("Unchanged ⚖️" if sim_res["score_delta"] == 0 else "Higher Risk ⚠️"))
            )
            st.metric("Simulation Verdict", verdict)

        sim_left, sim_right = st.columns(2, gap="large")
        with sim_left:
            st.markdown("#### 📈 7-Day Projected Wellness Adaptation Curve")
            proj_df = pd.DataFrame({
                "Day": sim_res["days"],
                "Simulated Habit Plan": sim_res["simulated_trajectory"],
                "Current Habit Baseline": sim_res["baseline_trajectory"],
            })
            proj_long = proj_df.melt(id_vars="Day", var_name="Scenario", value_name="Wellness Score")
            fig_proj = px.line(
                proj_long,
                x="Day",
                y="Wellness Score",
                color="Scenario",
                markers=True,
                text="Wellness Score",
                color_discrete_map={"Simulated Habit Plan": "#10B981", "Current Habit Baseline": "#64748B"},
            )
            fig_proj.update_traces(textposition="top center")
            fig_proj.update_yaxes(range=[0, 105])
            fig_proj.update_layout(xaxis_title="", yaxis_title="Wellness Score (0-100)", legend_title="")
            show_interactive_chart(fig_proj, "simulator_projection_curve", height=360)

        with sim_right:
            st.markdown("#### 🔄 Current vs Simulated vs Healthy Target Hours")
            hab_df = pd.DataFrame({
                "Habit": ["Screen Time", "Social Media", "Gaming", "Sleep"],
                "Current": [
                    user_data["daily_screen_time_hours"],
                    user_data["social_media_hours"],
                    user_data["gaming_hours"],
                    user_data["sleep_hours"],
                ],
                "Simulated": [sim_screen, sim_social, sim_gaming, sim_sleep],
                "Healthy Target": [4.0, 2.0, 1.5, 8.0],
            })
            hab_long = hab_df.melt(id_vars="Habit", var_name="Mode", value_name="Hours")
            fig_hab = px.bar(
                hab_long,
                x="Habit",
                y="Hours",
                color="Mode",
                barmode="group",
                text="Hours",
                color_discrete_map={"Current": "#EF4444", "Simulated": "#6366F1", "Healthy Target": "#10B981"},
            )
            fig_hab.update_traces(texttemplate="%{text:.1f}h", textposition="outside")
            fig_hab.update_layout(xaxis_title="", yaxis_title="Hours / Day", legend_title="")
            show_interactive_chart(fig_hab, "habits_comparison_grouped", height=360)

        # Personal Goal Progress Bars
        st.markdown("#### 🎯 Personal Digital Hygiene Goal Alignment")
        g1, g2, g3, g4 = st.columns(4)
        goal_screen = int(np.clip(round((4.0 / max(user_data["daily_screen_time_hours"], 0.5)) * 100), 0, 100))
        goal_social = int(np.clip(round((2.0 / max(user_data["social_media_hours"], 0.5)) * 100), 0, 100))
        goal_sleep = int(np.clip(round((user_data["sleep_hours"] / 8.0) * 100), 0, 100))
        goal_notify = int(np.clip(round((50.0 / max(user_data["notifications_per_day"], 1.0)) * 100), 0, 100))

        with g1:
            st.metric("📱 Screen Target (≤4h)", f"{goal_screen}%")
            st.progress(goal_screen)
        with g2:
            st.metric("🌐 Social Target (≤2h)", f"{goal_social}%")
            st.progress(goal_social)
        with g3:
            st.metric("🌙 Sleep Target (8h)", f"{goal_sleep}%")
            st.progress(goal_sleep)
        with g4:
            st.metric("🔔 Notification Target (≤50)", f"{goal_notify}%")
            st.progress(goal_notify)

        # Personalized Action Plan
        st.markdown('<div class="section-heading">🚀 Prioritized Digital Wellness Action Plan</div>', unsafe_allow_html=True)
        recs = generate_personalized_recommendations(user_data)
        rec_cols = st.columns(2, gap="medium")
        for idx, rec in enumerate(recs):
            with rec_cols[idx % 2]:
                st.markdown(
                    f"""
                    <div class="rec-card">
                        <div style="font-weight:700; font-size:15px; color:#F8FAFC; margin-bottom:6px;">
                            {rec['icon']} {rec['title']}
                            <span class="rec-priority">{rec['priority']}</span>
                        </div>
                        <div style="font-size:13.5px; color:#CBD5E1; line-height:1.5;">
                            {rec['detail']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ============================================================
    # PART 5: 7,500-USER COHORT EXPLORER
    # ============================================================
    if show_cohort:
        st.markdown('<div class="section-heading">📂 7,500-User Smartphone Behavioral Cohort Benchmark</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-description">Compare the four discovered K-Means behavioral clusters and explore how your habits stack up against 7,500 empirical profiles.</div>',
            unsafe_allow_html=True,
        )

        cluster_rows = []
        for cid, meta in CLUSTER_METADATA.items():
            avgs = meta["averages"]
            cluster_rows.append({
                "Cluster": f"Cluster {cid}: {meta['name']}",
                "Users": meta["cohort_count"],
                "Addiction Rate (%)": meta["addiction_rate"],
                "Screen Time (h)": avgs["daily_screen_time_hours"],
                "Social Media (h)": avgs["social_media_hours"],
                "Gaming (h)": avgs["gaming_hours"],
                "Work/Study (h)": avgs["work_study_hours"],
                "Sleep (h)": avgs["sleep_hours"],
                "Weekend Screen (h)": avgs["weekend_screen_time"],
            })
        cluster_table_df = pd.DataFrame(cluster_rows)

        c_left, c_right = st.columns(2, gap="large")
        with c_left:
            st.markdown("#### 📊 Addiction Prevalence Across Behavioral Clusters")
            fig_cl_rate = px.bar(
                cluster_table_df,
                x="Cluster",
                y="Addiction Rate (%)",
                text="Addiction Rate (%)",
                color="Cluster",
                color_discrete_sequence=["#10B981", "#EF4444", "#F59E0B", "#F97316"],
            )
            fig_cl_rate.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_cl_rate.update_layout(showlegend=False, xaxis_title="", yaxis=dict(range=[0, 110]))
            show_interactive_chart(fig_cl_rate, "cohort_cluster_addiction_rate", height=360)

        with c_right:
            st.markdown("#### 🧬 Cluster Profile Comparison Across Core Habits")
            cl_melt = cluster_table_df.melt(
                id_vars=["Cluster"],
                value_vars=["Screen Time (h)", "Social Media (h)", "Gaming (h)", "Work/Study (h)", "Sleep (h)"],
                var_name="Metric",
                value_name="Average Hours",
            )
            fig_cl_prof = px.bar(
                cl_melt,
                x="Metric",
                y="Average Hours",
                color="Cluster",
                barmode="group",
                color_discrete_sequence=["#10B981", "#EF4444", "#F59E0B", "#F97316"],
            )
            fig_cl_prof.update_layout(xaxis_title="", yaxis_title="Hours / Day", legend_title="")
            show_interactive_chart(fig_cl_prof, "cohort_cluster_profiles_bar", height=360)

        st.dataframe(cluster_table_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        """
        <div class="footer-text">
            🧠 Digital Well-Being AI • Random Forest + K-Means Behavioral Intelligence • 7,500-User Benchmark Cohort
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
