# =========================================
# Depot-wise Spot Absenteeism Training (XGB + GAN augmentation)
# Stable GAN (BCEWithLogitsLoss), batching, scaling,
# safe UTF-8 printing, and saves model+feature-list to avoid shape mismatch
# =========================================

import os
import sys
import json
import traceback
import warnings
import pickle

import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ---------------------------------
# Safe stdout (UTF-8) + safe_print
# ---------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def safe_print(*args, **kwargs):
    """Print without crashing on weird encodings."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(map(str, args))
        print(text.encode("ascii", errors="replace").decode(), **kwargs)


# -------------------
# MySQL Connection
# -------------------
# Preferred: pass MYSQL_URL from environment (e.g. from Streamlit app)
MYSQL_URL = os.getenv("MYSQL_URL")

if not MYSQL_URL:
    # Fallback: direct cloud MySQL creds (your freesqldatabase.com instance)
    MYSQL_USER = "sql12810449"
    MYSQL_PASS = "U1yWWaTs9n"          # <- your cloud password
    MYSQL_DB   = "sql12810449"
    MYSQL_HOST = "sql12.freesqldatabase.com"
    MYSQL_PORT = 3306

    MYSQL_URL = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )

safe_print("Using MySQL URL:", MYSQL_URL.replace(MYSQL_PASS, "******"))
engine = create_engine(MYSQL_URL, pool_pre_ping=True)

# -------------------
# Paths
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "xgb_models")
os.makedirs(SAVE_DIR, exist_ok=True)
METRICS_FILE = os.path.join(SAVE_DIR, "xgb_gan_metrics.json")

# -------------------
# Load Data
# -------------------
query = "SELECT * FROM input_data"
df = pd.read_sql(query, engine)

if "data_date" not in df.columns:
    raise KeyError("Column 'data_date' not found in input_data table.")
if "depot_name" not in df.columns:
    raise KeyError("Column 'depot_name' not found in input_data table.")
if "Spot_Absent" not in df.columns:
    raise KeyError("Column 'Spot_Absent' not found in input_data table.")

df["data_date"] = pd.to_datetime(df["data_date"])
df = df.sort_values(["depot_name", "data_date"]).reset_index(drop=True)


# -------------------
# Add lag features
# -------------------
def add_lag_features(df_in, target="Spot_Absent"):
    df_sorted = df_in.sort_values(["depot_name", "data_date"]).copy()
    df_sorted[f"{target}_Lag1"] = df_sorted.groupby("depot_name")[target].shift(1)
    df_sorted[f"{target}_MA3"] = (
        df_sorted.groupby("depot_name")[target].shift(1).rolling(3).mean()
    )
    return df_sorted


df = add_lag_features(df, target="Spot_Absent")

# -------------------
# GAN: Generator / Discriminator (BCEWithLogitsLoss)
# -------------------
class Generator(nn.Module):
    def __init__(self, noise_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),  # raw logits — use BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x)


# -------------------
# Stable GAN training & generation (mini-batches, scaling)
# -------------------
def generate_synthetic_data(X, n_samples=100, epochs=200, batch_size=32, debug=False):
    """
    X: pandas.DataFrame (numeric, no NaNs)
    returns: pandas.DataFrame of synthetic rows with same columns as X
    """
    if X.shape[0] < 4:
        raise ValueError("Not enough rows to train GAN reliably")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # scale to [0,1]
    scaler = MinMaxScaler()
    X_values = X.values.astype(float)
    X_scaled = scaler.fit_transform(X_values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    feature_dim = X.shape[1]
    noise_dim = max(8, feature_dim)

    G = Generator(noise_dim, feature_dim).to(device)
    D = Discriminator(feature_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    g_opt = optim.Adam(G.parameters(), lr=1e-3)
    d_opt = optim.Adam(D.parameters(), lr=1e-3)

    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(X)),
        shuffle=True,
        drop_last=False,
    )

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(epochs):
        for batch in loader:
            real_batch = batch[0].to(device)

            # Train Discriminator
            d_opt.zero_grad()
            logits_real = D(real_batch)
            labels_real = torch.full(
                (logits_real.size(0), 1),
                real_label,
                dtype=torch.float32,
                device=device,
            )
            loss_real = criterion(logits_real, labels_real)

            z = torch.randn(logits_real.size(0), noise_dim, device=device)
            fake_batch = G(z).detach()
            logits_fake = D(fake_batch)
            labels_fake = torch.full(
                (logits_fake.size(0), 1),
                fake_label,
                dtype=torch.float32,
                device=device,
            )
            loss_fake = criterion(logits_fake, labels_fake)

            d_loss = loss_real + loss_fake
            d_loss.backward()
            d_opt.step()

            # Train Generator
            g_opt.zero_grad()
            z = torch.randn(logits_real.size(0), noise_dim, device=device)
            gen_batch = G(z)
            gen_logits = D(gen_batch)
            labels_gen = torch.full(
                (gen_logits.size(0), 1),
                real_label,
                dtype=torch.float32,
                device=device,
            )
            g_loss = criterion(gen_logits, labels_gen)
            g_loss.backward()
            g_opt.step()

        if debug and (epoch % 50 == 0 or epoch == epochs - 1):
            with torch.no_grad():
                sample_z = torch.randn(min(32, len(X)), noise_dim, device=device)
                sample_fake = G(sample_z)
                d_r = D(real_batch).sigmoid()
                d_f = D(sample_fake).sigmoid()
                safe_print(
                    f"GAN epoch {epoch}: D(real) mean={d_r.mean().item():.3f}, "
                    f"D(fake) mean={d_f.mean().item():.3f}"
                )

    # generate synthetic samples
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, noise_dim, device=device)
        synthetic = G(z).cpu().numpy()

    # inverse scale
    synthetic_unscaled = scaler.inverse_transform(synthetic)
    synth_df = pd.DataFrame(synthetic_unscaled, columns=X.columns)
    return synth_df


# -------------------
# XGBoost training helper
# -------------------
def train_xgb_model(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        eval_metric="rmse",
    )
    try:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    except TypeError:
        model.fit(X_train, y_train)
    return model


# -------------------
# Train per-depot: GAN augmentation + XGB training
# -------------------
def train_depot_models(df_in, depot, target="Spot_Absent", augment_to=60, debug_gan=False):
    df_depot = df_in[df_in["depot_name"] == depot].dropna(subset=[target]).copy()
    n_rows = df_depot.shape[0]
    if n_rows < 8:
        safe_print(f"Skipping {depot} (too few rows: {n_rows})")
        return None, None

    # Candidate features (exclude id/time/target)
    features = [c for c in df_depot.columns if c not in ["depot_name", "data_date", target]]
    X = df_depot[features].copy()
    y = df_depot[target].copy()

    # numeric-only
    X = X.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if X.shape[1] == 0 or y.isna().all():
        safe_print(f"Skipping {depot} (no numeric features or missing target).")
        return None, None

    # drop rows with NaNs in selected features/target
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # Augment with synthetic data if small
    try:
        if len(X) < augment_to:
            n_synth = augment_to - len(X)
            safe_print(f"{depot}: generating {n_synth} synthetic rows (orig={len(X)})")
            synth_X = generate_synthetic_data(
                X,
                n_samples=n_synth,
                epochs=200,
                batch_size=min(32, len(X)),
                debug=debug_gan,
            )
            # simple target assignment: sample from historical target distribution
            synth_y = np.random.choice(y.values, size=n_synth, replace=True)
            X = pd.concat([X, synth_X], ignore_index=True)
            y = pd.concat([y, pd.Series(synth_y)], ignore_index=True)
            safe_print(f"{depot}: augmented -> new rows = {len(X)}")
    except Exception as e:
        safe_print(f"WARNING: GAN augmentation failed for {depot}: {e}")
        safe_print(traceback.format_exc())

    # Ensure we still have enough rows
    if len(X) < 8:
        safe_print(f"Skipping {depot} after augmentation (not enough usable rows).")
        return None, None

    # Train/test split — keep ordering (time-aware would be better)
    split_idx = int(np.floor(0.8 * len(X)))
    if split_idx < 2:
        safe_print(f"Skipping {depot} (not enough rows for split).")
        return None, None

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # Train model
    model = train_xgb_model(X_train.values, y_train.values, X_test.values, y_test.values)

    # Evaluate
    y_pred = model.predict(X_test.values)
    rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
    mae = mean_absolute_error(y_test.values, y_pred)
    r2 = r2_score(y_test.values, y_pred)

    metrics = {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "r2": round(r2, 3),
        "n_rows": n_rows,
        "features": list(X.columns),
    }

    # Save model + feature list together (using pickle instead of joblib)
    save_obj = {"model": model, "features": list(X.columns)}
    model_path = os.path.join(SAVE_DIR, f"{depot}_xgb.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(save_obj, f)

    return model, metrics


# -------------------
# Prediction helper (if ever used standalone)
# -------------------
def predict_depot_from_saved(depot, X_input_df):
    model_file = os.path.join(SAVE_DIR, f"{depot}_xgb.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Saved model not found for depot {depot}")
    with open(model_file, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    feat_list = saved["features"]

    # Align columns (fill missing with 0)
    X_aligned = X_input_df.reindex(columns=feat_list, fill_value=0)
    return model.predict(X_aligned.values)


# -------------------
# Main loop
# -------------------
all_metrics = {}
depots = df["depot_name"].unique()
safe_print(f"Starting training for {len(depots)} depots...")

for depot in depots:
    safe_print(f"\nTraining depot: {depot}")
    try:
        model, metrics = train_depot_models(
            df, depot, target="Spot_Absent", augment_to=60, debug_gan=False
        )
        if metrics:
            all_metrics[depot] = metrics
            safe_print(
                f"{depot}: R2={metrics['r2']} | RMSE={metrics['rmse']} | MAE={metrics['mae']}"
            )
        else:
            safe_print(f"Skipped {depot}")
    except Exception as e:
        safe_print(f"ERROR training {depot}: {e}")
        safe_print(traceback.format_exc())

# Save metrics file
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)

safe_print("\nTraining completed. Models and metrics saved in:", SAVE_DIR)
