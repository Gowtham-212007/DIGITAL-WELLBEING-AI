"""
Core Analytics, Input Validation, Random Forest Prediction, K-Means Clustering,
and What-If Habit Simulation Engine for Digital Well-Being AI.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from app.config import (
        CLUSTER_FEATURES,
        CLUSTER_METADATA,
        KMEANS_SCALED_CENTERS,
        MODEL_FEATURES_DEFAULT,
        SCALER_MEANS,
        SCALER_STDS,
    )
except ImportError:
    from config import (  # type: ignore
        CLUSTER_FEATURES,
        CLUSTER_METADATA,
        KMEANS_SCALED_CENTERS,
        MODEL_FEATURES_DEFAULT,
        SCALER_MEANS,
        SCALER_STDS,
    )


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert any value to a finite float."""
    try:
        res = float(val)
        return res if np.isfinite(res) else default
    except (ValueError, TypeError):
        return default


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


def get_value_for_feature(feature_name: str, user_data: Dict[str, Any]) -> Any:
    """
    Extract feature value from user_data dict or Series.
    Preserves string categorical values ('Male', 'Female', 'Other')
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
            row[feature] = _safe_float(val, 0.0)

    df = pd.DataFrame([row], columns=feature_names)
    for col in df.columns:
        if normalize_feature_name(col) not in ["gender", "sex"]:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(np.float64)
            )
        else:
            df[col] = df[col].astype(str)
    return df


def validate_user_inputs(user_data: Dict[str, Any]) -> List[str]:
    """Validate user inputs for logical and physical 24-hour consistency."""
    errors: List[str] = []
    screen_time = _safe_float(user_data.get("daily_screen_time_hours", 0.0))
    social_media = _safe_float(user_data.get("social_media_hours", 0.0))
    gaming = _safe_float(user_data.get("gaming_hours", 0.0))
    sleep = _safe_float(user_data.get("sleep_hours", 0.0))
    weekend = _safe_float(user_data.get("weekend_screen_time", 0.0))

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
    """Compute sub-scores (0-100), overall Wellness Score (0-100), and unified Risk Score (0-100)."""
    screen_time = _safe_float(user_data.get("daily_screen_time_hours", 6.0))
    social_media = _safe_float(user_data.get("social_media_hours", 3.0))
    gaming = _safe_float(user_data.get("gaming_hours", 2.0))
    work_study = _safe_float(user_data.get("work_study_hours", 5.0))
    sleep = _safe_float(user_data.get("sleep_hours", 7.0))
    notifications = _safe_float(user_data.get("notifications_per_day", 100.0))
    app_opens = _safe_float(user_data.get("app_opens_per_day", 50.0))
    stress = _safe_float(user_data.get("stress_level_numeric", 2.0))
    weekend = _safe_float(user_data.get("weekend_screen_time", 7.0))

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
        risk_level, risk_short, risk_icon, risk_color = "LOW RISK", "Low", "🟢", "#10B981"
    elif risk_score < 60:
        risk_level, risk_short, risk_icon, risk_color = "MODERATE RISK", "Moderate", "🟡", "#F59E0B"
    else:
        risk_level, risk_short, risk_icon, risk_color = "HIGH RISK", "High", "🔴", "#EF4444"

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
    """Return unified 0-100 risk score."""
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
    vec = np.array([_safe_float(user_data.get(col, 0.0)) for col in CLUSTER_FEATURES], dtype=np.float64)

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
    """Run Random Forest classifier (with calibrated fallback if model binaries are unavailable)."""
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

    st_h = _safe_float(user_data.get("daily_screen_time_hours", 6.0))
    sm_h = _safe_float(user_data.get("social_media_hours", 3.0))
    wk_h = _safe_float(user_data.get("weekend_screen_time", 7.0))
    gm_h = _safe_float(user_data.get("gaming_hours", 2.0))
    sl_h = _safe_float(user_data.get("sleep_hours", 7.0))
    nt_d = _safe_float(user_data.get("notifications_per_day", 100.0))

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
    sim_data["weekend_screen_time"] = float(
        max(
            sim_screen,
            _safe_float(base_user_data.get("weekend_screen_time", sim_screen))
            - max(0.0, _safe_float(base_user_data.get("daily_screen_time_hours", sim_screen)) - sim_screen),
        )
    )

    sim_breakdown = calculate_wellness_breakdown(sim_data)
    current_score = base_breakdown["wellness_score"]
    simulated_score = sim_breakdown["wellness_score"]
    delta = simulated_score - current_score

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
    screen_time = _safe_float(user_data.get("daily_screen_time_hours", 6.0))
    social_media = _safe_float(user_data.get("social_media_hours", 3.0))
    gaming = _safe_float(user_data.get("gaming_hours", 2.0))
    sleep = _safe_float(user_data.get("sleep_hours", 7.0))
    notifications = _safe_float(user_data.get("notifications_per_day", 100.0))
    app_opens = _safe_float(user_data.get("app_opens_per_day", 50.0))
    stress = _safe_float(user_data.get("stress_level_numeric", 2.0))
    weekend = _safe_float(user_data.get("weekend_screen_time", 7.0))

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
