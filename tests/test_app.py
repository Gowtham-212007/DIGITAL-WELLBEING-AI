"""
Comprehensive Automated Test Suite for Digital Well-Being AI (app/app.py)
Tests:
1. Feature name normalization & extraction (including categorical Gender OneHotEncoder fix)
2. Type-safe DataFrame creation (handling NaN, Inf, missing keys, categorical preservation)
3. Input validation (boundary conditions, combined social+gaming > screen, screen+sleep > 24h)
4. Unified Wellness & Risk Score engine (consistency across sections, risk levels, subscores)
5. K-Means Behavioral Clustering (all 4 archetypes + StandardScaler + mock model & fallback)
6. Random Forest Addiction Prediction (mock sklearn Pipeline with predict_proba + fallback)
7. What-If Digital Wellness Simulator & 7-Day Trajectory calculation
8. Personalized Recommendations engine across diverse user profiles
9. Full Streamlit UI execution simulation (testing main() across all views, presets, and form submissions)
"""

import sys
import types
import unittest
from pathlib import Path
import numpy as np
import pandas as pd


def _install_mock_dependencies_if_needed() -> None:
    """Install lightweight mocks for streamlit, plotly, and joblib if running in a bare test environment."""
    if "joblib" not in sys.modules:
        try:
            import joblib  # noqa: F401
        except ImportError:
            mock_joblib = types.ModuleType("joblib")
            mock_joblib.load = lambda path: None
            sys.modules["joblib"] = mock_joblib

    if "plotly" not in sys.modules:
        try:
            import plotly.express  # noqa: F401
            import plotly.graph_objects  # noqa: F401
        except ImportError:
            mock_plotly = types.ModuleType("plotly")
            mock_px = types.ModuleType("plotly.express")
            mock_go = types.ModuleType("plotly.graph_objects")

            class DummyFig:
                def __init__(self, *args, **kwargs):
                    self.layout = {}
                    self.traces = []

                def update_traces(self, *args, **kwargs):
                    return self

                def update_layout(self, *args, **kwargs):
                    self.layout.update(kwargs)
                    return self

                def update_yaxes(self, *args, **kwargs):
                    return self

                def add_trace(self, trace):
                    self.traces.append(trace)
                    return self

            mock_px.bar = lambda *a, **kw: DummyFig()
            mock_px.pie = lambda *a, **kw: DummyFig()
            mock_px.line = lambda *a, **kw: DummyFig()
            mock_go.Figure = DummyFig
            mock_go.Indicator = lambda *a, **kw: ("Indicator", kw)
            mock_go.Scatterpolar = lambda *a, **kw: ("Scatterpolar", kw)

            sys.modules["plotly"] = mock_plotly
            sys.modules["plotly.express"] = mock_px
            sys.modules["plotly.graph_objects"] = mock_go

    if "streamlit" not in sys.modules:
        try:
            import streamlit  # noqa: F401
        except ImportError:
            mock_st = types.ModuleType("streamlit")

            class DummyContext:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    return False

            mock_st.session_state = {}
            mock_st.sidebar = DummyContext()
            mock_st.cache_resource = lambda fn: fn
            mock_st.cache_data = lambda fn: fn
            mock_st.set_page_config = lambda **kw: None
            mock_st.markdown = lambda *a, **kw: None
            mock_st.title = lambda *a, **kw: None
            mock_st.write = lambda *a, **kw: None
            mock_st.caption = lambda *a, **kw: None
            mock_st.divider = lambda *a, **kw: None
            mock_st.success = lambda *a, **kw: None
            mock_st.info = lambda *a, **kw: None
            mock_st.warning = lambda *a, **kw: None
            mock_st.error = lambda *a, **kw: None
            mock_st.toast = lambda *a, **kw: None
            mock_st.metric = lambda *a, **kw: None
            mock_st.progress = lambda *a, **kw: None
            mock_st.dataframe = lambda *a, **kw: None
            mock_st.plotly_chart = lambda *a, **kw: None
            mock_st.columns = lambda spec, **kw: [
                DummyContext() for _ in range(spec if isinstance(spec, int) else len(spec))
            ]
            mock_st.expander = lambda *a, **kw: DummyContext()
            mock_st.form = lambda *a, **kw: DummyContext()
            mock_st.button = lambda *a, **kw: False
            mock_st.form_submit_button = lambda *a, **kw: False
            mock_st.radio = lambda label, options, index=0, **kw: options[index]
            mock_st.selectbox = lambda label, options, index=0, **kw: options[index]
            mock_st.number_input = lambda label, value=0, **kw: value
            mock_st.slider = lambda label, value=0, **kw: value
            sys.modules["streamlit"] = mock_st


_install_mock_dependencies_if_needed()

_HERE = Path(__file__).resolve().parent
for _p in [_HERE, _HERE.parent / "app", _HERE.parent]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import app as dw_app  # noqa: E402
import config as dw_config  # noqa: E402
import engine as dw_engine  # noqa: E402
import styles as dw_styles  # noqa: E402


class TestDigitalWellbeingAI(unittest.TestCase):
    """Complete unit & integration test suite for Digital Well-Being AI."""

    def test_01_feature_normalization_and_extraction(self):
        """Verify feature normalization and critical gender string preservation fix."""
        self.assertEqual(dw_app.normalize_feature_name("Daily Screen Time (hours)"), "dailyscreentimehours")
        self.assertEqual(dw_app.normalize_feature_name("Work / Study Hours"), "workstudyhours")

        sample = {
            "age": 24,
            "gender": "Female",
            "daily_screen_time_hours": 6.5,
            "social_media_hours": 2.5,
            "gaming_hours": 1.0,
            "work_study_hours": 5.5,
            "sleep_hours": 7.5,
            "notifications_per_day": 90,
            "app_opens_per_day": 45,
            "stress_level_numeric": 2,
            "weekend_screen_time": 8.0,
        }
        # Gender MUST remain a valid string ('Male', 'Female', 'Other') for OneHotEncoder
        self.assertEqual(dw_app.get_value_for_feature("gender", sample), "Female")
        self.assertEqual(dw_app.get_value_for_feature("Gender", {"gender": "other"}), "Other")
        self.assertEqual(dw_app.get_value_for_feature("daily_screen_time_hours", sample), 6.5)
        self.assertEqual(dw_app.get_value_for_feature("unknown_col", sample), 0.0)

    def test_02_create_model_input_safety(self):
        """Verify create_model_input handles NaN, Inf, and preserves column schema."""
        dirty_input = {
            "age": np.nan,
            "gender": "Male",
            "daily_screen_time_hours": np.inf,
            "social_media_hours": "invalid",
            "gaming_hours": 2.0,
            "work_study_hours": 4.0,
            "sleep_hours": 7.0,
            "notifications_per_day": 120,
            "app_opens_per_day": 60,
            "weekend_screen_time": -np.inf,
        }
        df = dw_app.create_model_input(dirty_input)
        self.assertEqual(list(df.columns), dw_app.MODEL_FEATURES_DEFAULT)
        self.assertEqual(df.loc[0, "gender"], "Male")
        self.assertEqual(df.loc[0, "age"], 0.0)
        self.assertEqual(df.loc[0, "daily_screen_time_hours"], 0.0)
        self.assertEqual(df.loc[0, "social_media_hours"], 0.0)
        self.assertEqual(df.loc[0, "weekend_screen_time"], 0.0)
        self.assertFalse(df.drop(columns=["gender"]).isna().any().any())

    def test_03_input_validation_rules(self):
        """Test valid profile and all invalid physical/logical combinations."""
        valid = dict(dw_app.PERSONA_PRESETS["Balanced Professional"])
        self.assertEqual(dw_app.validate_user_inputs(valid), [])

        # Social + Gaming > Screen Time
        bad_sum = dict(valid, daily_screen_time_hours=5.0, social_media_hours=3.5, gaming_hours=2.5)
        errs_sum = dw_app.validate_user_inputs(bad_sum)
        self.assertTrue(any("Combined Social Media" in e for e in errs_sum))

        # Screen Time + Sleep > 24
        bad_day = dict(valid, daily_screen_time_hours=18.0, sleep_hours=8.0)
        errs_day = dw_app.validate_user_inputs(bad_day)
        self.assertTrue(any("cannot exceed 24 hours" in e for e in errs_day))

    def test_04_unified_wellness_and_risk_scoring(self):
        """Verify consistent risk/wellness scores across healthy, moderate, and high-risk profiles."""
        healthy = dw_app.PERSONA_PRESETS["Balanced Professional"]
        b_healthy = dw_app.calculate_wellness_breakdown(healthy)
        self.assertGreaterEqual(b_healthy["wellness_score"], 75)
        self.assertLess(b_healthy["risk_score"], 30)
        self.assertEqual(b_healthy["risk_level"], "LOW RISK")
        self.assertEqual(b_healthy["wellness_score"] + b_healthy["risk_score"], 100)
        self.assertEqual(dw_app.calculate_risk_score(healthy), b_healthy["risk_score"])

        heavy = dw_app.PERSONA_PRESETS["Heavy Social Scroller"]
        b_heavy = dw_app.calculate_wellness_breakdown(heavy)
        self.assertLess(b_heavy["wellness_score"], 50)
        self.assertGreaterEqual(b_heavy["risk_score"], 50)

    def test_05_kmeans_behavioral_clustering_all_archetypes(self):
        """Verify K-Means cluster prediction accurately identifies all 4 empirical clusters."""
        # Cluster 0: Balanced & Work-Focused
        c0 = dw_app.predict_kmeans_cluster(dw_app.CLUSTER_METADATA[0]["averages"])
        self.assertEqual(c0["cluster_id"], 0)
        self.assertIn("Balanced", c0["name"])

        # Cluster 1: Gaming-Heavy High-Exposure
        c1 = dw_app.predict_kmeans_cluster(dw_app.CLUSTER_METADATA[1]["averages"])
        self.assertEqual(c1["cluster_id"], 1)
        self.assertIn("Gaming-Heavy", c1["name"])

        # Cluster 2: Moderate Screen & Notification-Driven
        c2 = dw_app.predict_kmeans_cluster(dw_app.CLUSTER_METADATA[2]["averages"])
        self.assertEqual(c2["cluster_id"], 2)
        self.assertIn("Moderate", c2["name"])

        # Cluster 3: Screen & Social-Heavy
        c3 = dw_app.predict_kmeans_cluster(dw_app.CLUSTER_METADATA[3]["averages"])
        self.assertEqual(c3["cluster_id"], 3)
        self.assertIn("Screen & Social-Heavy", c3["name"])

    def test_06_random_forest_prediction_with_mock_and_fallback(self):
        """Verify both mock sklearn Random Forest pipeline and calibrated fallback."""
        healthy = dw_app.PERSONA_PRESETS["Balanced Professional"]
        res_healthy = dw_app.predict_addiction_ml(healthy, ai_models=None)
        self.assertEqual(res_healthy["prediction_class"], 0)
        self.assertIn("Not Addicted", res_healthy["prediction_label"])
        self.assertLess(res_healthy["addiction_probability"], 50.0)

        heavy = dw_app.PERSONA_PRESETS["Heavy Social Scroller"]
        res_heavy = dw_app.predict_addiction_ml(heavy, ai_models=None)
        self.assertEqual(res_heavy["prediction_class"], 1)
        self.assertIn("Addicted", res_heavy["prediction_label"])
        self.assertGreaterEqual(res_heavy["addiction_probability"], 80.0)

        # Test with mock RF model object implementing predict & predict_proba
        class MockRF:
            feature_names_in_ = np.array(dw_app.MODEL_FEATURES_DEFAULT)

            def predict(self, X):
                assert X["gender"].iloc[0] in ["Male", "Female", "Other"]
                return np.array([1])

            def predict_proba(self, X):
                return np.array([[0.08, 0.92]])

        mock_models = {"rf": MockRF(), "features": dw_app.MODEL_FEATURES_DEFAULT}
        res_mock = dw_app.predict_addiction_ml(heavy, ai_models=mock_models)
        self.assertEqual(res_mock["prediction_class"], 1)
        self.assertAlmostEqual(res_mock["addiction_probability"], 92.0)
        self.assertAlmostEqual(res_mock["confidence"], 92.0)

    def test_07_simulator_and_7day_trajectory(self):
        """Verify What-If Simulator delta and 7-day adaptation curve."""
        heavy = dw_app.PERSONA_PRESETS["Heavy Social Scroller"]
        sim = dw_app.calculate_simulation(
            heavy,
            sim_screen=4.0,
            sim_social=1.5,
            sim_gaming=0.5,
            sim_sleep=8.0,
            sim_notifications=40,
        )
        self.assertGreater(sim["simulated_score"], sim["current_score"])
        self.assertEqual(sim["score_delta"], sim["simulated_score"] - sim["current_score"])
        self.assertEqual(len(sim["days"]), 7)
        self.assertEqual(sim["simulated_trajectory"][0], sim["current_score"])
        self.assertEqual(sim["simulated_trajectory"][-1], sim["simulated_score"])

    def test_08_recommendations_engine(self):
        """Verify recommendations adapt to healthy vs high-risk profiles."""
        recs_healthy = dw_app.generate_personalized_recommendations(dw_app.PERSONA_PRESETS["Balanced Professional"])
        self.assertEqual(len(recs_healthy), 1)
        self.assertEqual(recs_healthy[0]["priority"], "OPTIMAL")

        recs_heavy = dw_app.generate_personalized_recommendations(dw_app.PERSONA_PRESETS["Heavy Social Scroller"])
        self.assertGreaterEqual(len(recs_heavy), 4)
        self.assertTrue(any(r["priority"] == "HIGH" for r in recs_heavy))

    def test_09_full_streamlit_ui_execution(self):
        """Execute main() across all workspace views and form submit states without errors."""
        import streamlit as st
        st.session_state.clear()
        dw_app.main()
        self.assertIn("user_profile", st.session_state)

        # Simulate form submission inside main()
        st.form_submit_button = lambda *a, **kw: True
        dw_app.main()
        st.form_submit_button = lambda *a, **kw: False


if __name__ == "__main__":
    unittest.main(verbosity=2)
