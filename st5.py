# app.py
import os
import json
import sys
import subprocess

import joblib
import streamlit as st
from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# -------------------
# Config / Paths
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV used instead of MySQL
DATA_CSV_PATH = os.path.join(BASE_DIR, "input_datasql.csv")

# Training script path (relative -> works locally & on Streamlit Cloud)
TRAIN_SCRIPT_PATH = os.path.join(BASE_DIR, "tr5.py")

SAVE_DIR = os.path.join(BASE_DIR, "xgb_models")
os.makedirs(SAVE_DIR, exist_ok=True)
METRICS_FILE = os.path.join(SAVE_DIR, "xgb_gan_metrics.json")

# -------------------
# Utility helpers
# -------------------
@st.cache_data
def load_input_data():
    """Load full input dataset from CSV (input_datasql.csv)."""
    if not os.path.exists(DATA_CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {DATA_CSV_PATH}")
    df = pd.read_csv(DATA_CSV_PATH)
    # Ensure data_date is datetime if present
    if "data_date" in df.columns:
        df["data_date"] = pd.to_datetime(df["data_date"])
    return df


@st.cache_data
def load_metrics():
    """Load metrics JSON if available."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_models_for_depot(depot):
    """
    Load joblib-saved model object for a depot.
    Support two save styles:
     1) joblib saved dict: {"model": model_obj, "features": [...]}
     2) joblib saved model only + separate features json file
    Returns (model_obj, feature_list) or (None, None)
    """
    model_path = os.path.join(SAVE_DIR, f"{depot}_xgb.pkl")
    feature_path = os.path.join(SAVE_DIR, f"{depot}_features.json")

    if not os.path.exists(model_path):
        return None, None

    try:
        loaded = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model for {depot}: {e}")
        return None, None

    # Case 1: dict container
    if isinstance(loaded, dict) and "model" in loaded and "features" in loaded:
        return loaded["model"], loaded["features"]

    # Case 2: direct model object + separate features json
    features = None
    if os.path.exists(feature_path):
        try:
            with open(feature_path, "r", encoding="utf-8") as f:
                features = json.load(f)
        except Exception:
            features = None

    return loaded, features


def predict_with_model(model, X_df):
    """
    Predict handling both XGBRegressor and xgboost.Booster (raw).
    Expects X_df as a pandas DataFrame already aligned to the training features.
    """
    if model is None:
        raise ValueError("Model is None")

    # If model is a dict wrapper (old style), extract
    if isinstance(model, dict) and "model" in model:
        model = model["model"]

    # If sklearn-style estimator (has predict)
    if hasattr(model, "predict"):
        return model.predict(X_df)
    # If raw xgboost Booster
    if isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X_df)
        return model.predict(dmat)
    # Unknown type
    raise TypeError(f"Unsupported model type: {type(model)}")


def align_row_to_features(row_dict, features):
    """
    Turn a single-row dict into DataFrame aligned to features.
    Missing features will be filled with 0.
    """
    df_row = pd.DataFrame([row_dict])
    df_row = df_row.reindex(columns=features, fill_value=0)
    return df_row


# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Depot Absenteeism ML", layout="wide")
st.title("🚍 Depot-wise Absenteeism Prediction (XGB + GAN)")

tab1, tab2, tab3 = st.tabs(["📊 Train Models", "🔮 Predict Absenteeism", "📈 Actual vs Predicted Analysis"])

# ------------------- Tab 1: Train / Retrain -------------------
with tab1:
    st.subheader("Depot-wise Training (runs your tr5.py)")
    st.write("This will execute the training script (tr5.py). It now uses input_datasql.csv instead of MySQL.")

    if st.button("Train / Retrain All Depots"):
        with st.spinner("Training depot-specific models — this may take several minutes..."):
            try:
                # Use the SAME Python interpreter Streamlit is running on
                result = subprocess.run(
                    [sys.executable, TRAIN_SCRIPT_PATH],
                    capture_output=True,
                    text=True,
                    check=False,  # we handle returncode manually below
                )
                if result.returncode == 0:
                    st.success("Training finished successfully.")
                    st.code(result.stdout[:10000])  # show first part
                else:
                    st.error("Training failed. See stderr below.")
                    st.code(result.stderr[:10000])
            except Exception as e:
                st.error("Failed to run training script:")
                st.code(str(e))

    metrics = load_metrics()
    if metrics:
        st.markdown("### Last training metrics (summary)")
        for depot, m in metrics.items():
            st.markdown(f"**{depot}**")
            cols = st.columns(3)
            keys = list(m.keys())
            for i, key in enumerate(keys):
                with cols[i % 3]:
                    val = m[key]
                    if isinstance(val, list):
                        display_val = str(val)[:150]
                    elif val is None:
                        display_val = "N/A"
                    else:
                        display_val = val
                    st.metric(label=key.upper(), value=display_val)
    else:
        st.info("No metrics found. Run training first (use the button above).")


# ------------------- Tab 2: Predict -------------------
with tab2:
    st.subheader("Predict Absenteeism for a Depot")

    try:
        df_all = load_input_data()
        if "depot_name" not in df_all.columns:
            raise KeyError("Column 'depot_name' not found in input_datasql.csv")

        depot_list = sorted(df_all["depot_name"].dropna().unique().tolist())
    except Exception as e:
        depot_list = []
        st.error(f"Failed to load depots from CSV: {e}")

    if not depot_list:
        st.info("No depots found in CSV. Make sure 'depot_name' column exists in input_datasql.csv.")
    else:
        selected_depot = st.selectbox("Select Depot", depot_list)
        selected_date = st.date_input("Select Date", date.today() + timedelta(days=1))

        if st.button("Predict Tomorrow's Absenteeism"):
            try:
                df_depot = df_all[df_all["depot_name"] == selected_depot].copy()
                if "data_date" in df_depot.columns:
                    df_depot = df_depot.sort_values("data_date", ascending=False)

                if df_depot.empty:
                    st.warning("No historical record found for selected depot in CSV.")
                else:
                    last_row = df_depot.iloc[0].to_dict()
                    last_row["data_date"] = pd.to_datetime(selected_date)

                    model, features = load_models_for_depot(selected_depot)
                    if model is None:
                        st.error("No trained model found for this depot. Please train first.")
                    else:
                        # if features list missing, try to infer from model saved dict
                        model_path = os.path.join(SAVE_DIR, f"{selected_depot}_xgb.pkl")
                        if features is None and os.path.exists(model_path):
                            loaded = joblib.load(model_path)
                            if isinstance(loaded, dict):
                                features = loaded.get("features", None)

                        if features is None:
                            st.error("Feature list for this depot not found. Can't make prediction.")
                        else:
                            X_row = align_row_to_features(last_row, features)
                            y_pred = predict_with_model(model, X_row)
                            st.success(
                                f"Predicted Spot Absent for {selected_depot} on {selected_date}: "
                                f"**{float(y_pred[0]):.2f}**"
                            )
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ------------------- Tab 3: Analysis -------------------
with tab3:
    st.subheader("Actual vs Predicted Analysis")

    try:
        df_all = load_input_data()
        if "depot_name" not in df_all.columns:
            raise KeyError("Column 'depot_name' not found in input_datasql.csv")

        depot_list = sorted(df_all["depot_name"].dropna().unique().tolist())
    except Exception as e:
        depot_list = []
        st.error(f"Failed to load depots from CSV: {e}")

    if not depot_list:
        st.info("No depots found in CSV.")
    else:
        selected_depot = st.selectbox("Select Depot for Analysis", depot_list, key="analysis_depot")

        # Filter for selected depot
        df_depot = df_all[df_all["depot_name"] == selected_depot].copy()

        # date range selection (fetch min/max from CSV)
        if "data_date" in df_depot.columns and not df_depot.empty:
            min_date = df_depot["data_date"].min().date()
            max_date = df_depot["data_date"].max().date()
        else:
            min_date = date.today() - timedelta(days=90)
            max_date = date.today()

        start_date = st.date_input("Start Date", min_date, key="analysis_start")
        end_date = st.date_input("End Date", max_date, key="analysis_end")

        if st.button("Run Analysis"):
            try:
                # Filter by date range (if data_date column exists)
                df = df_depot.copy()
                if "data_date" in df.columns:
                    df = df[(df["data_date"].dt.date >= start_date) & (df["data_date"].dt.date <= end_date)]
                    df = df.sort_values("data_date")

                if df.empty:
                    st.warning("No rows in selected range for this depot.")
                else:
                    model, features = load_models_for_depot(selected_depot)
                    if model is None:
                        st.error("Model for this depot not found. Train first.")
                    elif features is None:
                        st.error("Feature list for this depot not found. Can't run analysis.")
                    else:
                        df_features = df.reindex(columns=features, fill_value=0)
                        preds = predict_with_model(model, df_features)
                        df = df.copy()
                        df["Predicted"] = preds

                        if "Spot_Absent" not in df.columns:
                            st.error("Column 'Spot_Absent' not found in CSV; cannot compute errors.")
                        else:
                            df["Error"] = df["Spot_Absent"] - df["Predicted"]
                            df["Abs_Error"] = df["Error"].abs()
                            df["Pct_Error"] = (
                                df["Abs_Error"] / df["Spot_Absent"].replace(0, 1)
                            ) * 100

                            rmse = np.sqrt((df["Error"] ** 2).mean())
                            mae = df["Abs_Error"].mean()
                            mape = df["Pct_Error"].mean()
                            r2 = r2_score(df["Spot_Absent"], df["Predicted"])

                            st.markdown(
                                f"**RMSE:** {rmse:.2f}  |  "
                                f"**MAE:** {mae:.2f}  |  "
                                f"**MAPE:** {mape:.2f}%  |  "
                                f"**R²:** {r2:.2f}"
                            )

                            st.dataframe(df[["data_date", "Spot_Absent", "Predicted", "Error", "Pct_Error"]])

                            # Scatter
                            fig1, ax1 = plt.subplots()
                            sns.scatterplot(x="Spot_Absent", y="Predicted", data=df, ax=ax1)
                            ax1.plot(
                                [df["Spot_Absent"].min(), df["Spot_Absent"].max()],
                                [df["Spot_Absent"].min(), df["Spot_Absent"].max()],
                                "r--",
                            )
                            ax1.set_xlabel("Actual")
                            ax1.set_ylabel("Predicted")
                            st.pyplot(fig1)

                            # Error hist
                            fig2, ax2 = plt.subplots()
                            sns.histplot(df["Error"], bins=20, kde=True, ax=ax2)
                            st.pyplot(fig2)

                            # Time-series
                            fig3, ax3 = plt.subplots(figsize=(10, 4))
                            ax3.plot(df["data_date"], df["Spot_Absent"], label="Actual", marker="o")
                            ax3.plot(df["data_date"], df["Predicted"], label="Predicted", marker="x")
                            ax3.legend()
                            ax3.set_xlabel("Date")
                            ax3.set_ylabel("Spot Absent")
                            st.pyplot(fig3)

                            # allow download
                            csv = df[["data_date", "Spot_Absent", "Predicted", "Error", "Pct_Error"]].to_csv(index=False)
                            st.download_button(
                                "Download Results CSV",
                                csv,
                                file_name=f"{selected_depot}_analysis.csv",
                                mime="text/csv",
                            )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
