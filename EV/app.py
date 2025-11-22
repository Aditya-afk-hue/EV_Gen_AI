import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Gemini SDK (for chatbot only)
from google import genai


# ==============================
# 0. BASIC CONFIG
# ==============================
st.set_page_config(
    page_title="EV Cost Predictor & Gen-AI Studio",
    layout="wide",
)

DATA_PATH = Path("ev_feature_engineered.csv")  # CSV next to app.py


# ==============================
# 1. DATA & MODEL UTILITIES
# ==============================

@st.cache_data(show_spinner=True)
def load_data(path: Path) -> pd.DataFrame:
    """Load the feature-engineered EV dataset."""
    if not path.exists():
        st.error(f"CSV not found at {path.resolve()}")
        st.stop()
    df = pd.read_csv(path, low_memory=False)

    # Basic numeric coercion
    for c in [
        "base_msrp",
        "base_msrp_capped",
        "electric_range",
        "electric_range_capped",
        "vehicle_age",
        "price_per_mile",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing target price for price model
    df = df.dropna(subset=["base_msrp"])
    return df


# ---------- PRICE REGRESSOR ----------
@st.cache_resource(show_spinner=True)
def train_price_regressor(df: pd.DataFrame):
    """Train RandomForestRegressor to predict base_msrp."""
    numeric_feats = [
        c
        for c in [
            "electric_range_capped",
            "electric_range",
            "vehicle_age",
            "price_per_mile",
        ]
        if c in df.columns
    ]

    df = df.copy()
    cat_feats = []
    if "make" in df.columns:
        top_n = 20
        top_makes = df["make"].value_counts().head(top_n).index.tolist()
        df["make_top"] = df["make"].where(df["make"].isin(top_makes), other="OTHER")
        cat_feats.append("make_top")
    for c in ["range_group", "electric_vehicle_type"]:
        if c in df.columns:
            cat_feats.append(c)

    if numeric_feats:
        X_num = df[numeric_feats].copy().fillna(df[numeric_feats].median())
    else:
        X_num = pd.DataFrame(index=df.index)

    X_cat = pd.DataFrame(index=df.index)
    ohe = None
    if cat_feats:
        try:
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_cat_arr = ohe.fit_transform(df[cat_feats].fillna("NA").astype(str))
            cols = []
            for feat, cats in zip(cat_feats, ohe.categories_):
                for val in cats:
                    cols.append(f"{feat}__{val}")
            X_cat = pd.DataFrame(X_cat_arr, index=df.index, columns=cols)
        except Exception:
            X_cat = pd.get_dummies(
                df[cat_feats].fillna("NA").astype(str),
                prefix_sep="__",
                columns=cat_feats,
            )
            X_cat.index = df.index

    if not X_cat.empty and not X_num.empty:
        X = pd.concat(
            [X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1
        )
    elif not X_num.empty:
        X = X_num.reset_index(drop=True)
    else:
        X = X_cat.reset_index(drop=True)

    y = df["base_msrp"].reset_index(drop=True)
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    num_cols_in_X = X.select_dtypes(include=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    if num_cols_in_X:
        X_train[num_cols_in_X] = scaler.fit_transform(X_train[num_cols_in_X])
        X_test[num_cols_in_X] = scaler.transform(X_test[num_cols_in_X])

    model = RandomForestRegressor(
        n_estimators=120, max_depth=14, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    return model, scaler, ohe, cat_feats, numeric_feats, metrics, feature_names


# ---------- RANGE REGRESSOR ----------
@st.cache_resource(show_spinner=True)
def train_range_regressor(df: pd.DataFrame):
    """Train model to predict electric range (forecast range)."""
    df_r = df.copy()
    target_col = (
        "electric_range_capped"
        if "electric_range_capped" in df_r.columns
        else "electric_range"
    )
    df_r = df_r.dropna(subset=[target_col])

    numeric_feats = [
        c
        for c in [
            "base_msrp_capped",
            "base_msrp",
            "vehicle_age",
            "price_per_mile",
        ]
        if c in df_r.columns
    ]

    cat_feats = []
    if "make" in df_r.columns:
        top_n = 20
        top_makes = df_r["make"].value_counts().head(top_n).index.tolist()
        df_r["make_top"] = df_r["make"].where(df_r["make"].isin(top_makes), other="OTHER")
        cat_feats.append("make_top")
    for c in ["range_group", "electric_vehicle_type"]:
        if c in df_r.columns:
            cat_feats.append(c)

    if numeric_feats:
        X_num = df_r[numeric_feats].copy().fillna(df_r[numeric_feats].median())
    else:
        X_num = pd.DataFrame(index=df_r.index)

    X_cat = pd.DataFrame(index=df_r.index)
    ohe = None
    if cat_feats:
        try:
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_cat_arr = ohe.fit_transform(df_r[cat_feats].fillna("NA").astype(str))
            cols = []
            for feat, cats in zip(cat_feats, ohe.categories_):
                for val in cats:
                    cols.append(f"{feat}__{val}")
            X_cat = pd.DataFrame(X_cat_arr, index=df_r.index, columns=cols)
        except Exception:
            X_cat = pd.get_dummies(
                df_r[cat_feats].fillna("NA").astype(str),
                prefix_sep="__",
                columns=cat_feats,
            )
            X_cat.index = df_r.index

    if not X_cat.empty and not X_num.empty:
        X = pd.concat(
            [X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1
        )
    elif not X_num.empty:
        X = X_num.reset_index(drop=True)
    else:
        X = X_cat.reset_index(drop=True)

    y = df_r[target_col].reset_index(drop=True)
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    num_cols_in_X = X.select_dtypes(include=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    if num_cols_in_X:
        X_train[num_cols_in_X] = scaler.fit_transform(X_train[num_cols_in_X])
        X_test[num_cols_in_X] = scaler.transform(X_test[num_cols_in_X])

    model = RandomForestRegressor(
        n_estimators=120, max_depth=14, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    return model, scaler, ohe, cat_feats, numeric_feats, metrics, feature_names, target_col


# ---------- CAFV ELIGIBILITY CLASSIFIER ----------
@st.cache_resource(show_spinner=True)
def train_cafv_classifier(df: pd.DataFrame):
    """
    Train RandomForestClassifier to predict CAFV eligibility.
    Binary label: 1 = 'Clean Alternative Fuel Vehicle Eligible', 0 = all others.
    """
    target_col = "clean_alternative_fuel_vehicle_cafv_eligibility"
    if target_col not in df.columns:
        st.warning(f"{target_col} not found for classification.")
        return None, None, None, [], [], {}

    df_c = df.copy()
    df_c[target_col] = df_c[target_col].astype(str)

    df_c["cafv_label"] = np.where(
        df_c[target_col].str.contains("Clean Alternative Fuel Vehicle Eligible", case=False),
        1,
        0,
    )
    df_c = df_c.dropna(subset=["cafv_label"])

    numeric_feats = [
        c
        for c in [
            "electric_range_capped",
            "electric_range",
            "vehicle_age",
            "price_per_mile",
        ]
        if c in df_c.columns
    ]

    cat_feats = []
    if "make" in df_c.columns:
        top_n = 20
        top_makes = df_c["make"].value_counts().head(top_n).index.tolist()
        df_c["make_top"] = df_c["make"].where(df_c["make"].isin(top_makes), other="OTHER")
        cat_feats.append("make_top")
    for c in ["range_group", "electric_vehicle_type"]:
        if c in df_c.columns:
            cat_feats.append(c)

    if numeric_feats:
        X_num = df_c[numeric_feats].copy().fillna(df_c[numeric_feats].median())
    else:
        X_num = pd.DataFrame(index=df_c.index)

    X_cat = pd.DataFrame(index=df_c.index)
    ohe = None
    if cat_feats:
        try:
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_cat_arr = ohe.fit_transform(df_c[cat_feats].fillna("NA").astype(str))
            cols = []
            for feat, cats in zip(cat_feats, ohe.categories_):
                for val in cats:
                    cols.append(f"{feat}__{val}")
            X_cat = pd.DataFrame(X_cat_arr, index=df_c.index, columns=cols)
        except Exception:
            X_cat = pd.get_dummies(
                df_c[cat_feats].fillna("NA").astype(str),
                prefix_sep="__",
                columns=cat_feats,
            )
            X_cat.index = df_c.index

    if not X_cat.empty and not X_num.empty:
        X = pd.concat(
            [X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1
        )
    elif not X_num.empty:
        X = X_num.reset_index(drop=True)
    else:
        X = X_cat.reset_index(drop=True)

    y = df_c["cafv_label"].reset_index(drop=True)
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    num_cols_in_X = X.select_dtypes(include=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )

    if num_cols_in_X:
        X_train[num_cols_in_X] = scaler.fit_transform(X_train[num_cols_in_X])
        X_test[num_cols_in_X] = scaler.transform(X_test[num_cols_in_X])

    clf = RandomForestClassifier(
        n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }

    return clf, scaler, ohe, cat_feats, numeric_feats, metrics, feature_names


def prepare_single_example(
    df: pd.DataFrame,
    ohe,
    cat_feats,
    num_feats,
    scaler,
    make: str,
    ev_type: str,
    range_group: str,
    electric_range: float,
    vehicle_age: int,
    price_per_mile: float,
) -> pd.DataFrame:
    """Build a 1-row feature matrix matching training features."""
    row = {}

    # numeric
    if "electric_range_capped" in num_feats:
        row["electric_range_capped"] = min(electric_range, 500)
    if "electric_range" in num_feats:
        row["electric_range"] = electric_range
    if "vehicle_age" in num_feats:
        row["vehicle_age"] = vehicle_age
    if "price_per_mile" in num_feats:
        row["price_per_mile"] = price_per_mile
    if "base_msrp_capped" in num_feats:
        row["base_msrp_capped"] = price_per_mile * electric_range
    if "base_msrp" in num_feats:
        row["base_msrp"] = price_per_mile * electric_range

    # top makes
    df_local = df.copy()
    top_makes = []
    if "make" in df_local.columns:
        top_makes = df_local["make"].value_counts().head(20).index.tolist()

    # categorical
    if "make_top" in cat_feats:
        row["make_top"] = make if make in top_makes else "OTHER"
    if "range_group" in cat_feats:
        row["range_group"] = range_group
    if "electric_vehicle_type" in cat_feats:
        row["electric_vehicle_type"] = ev_type

    row_df = pd.DataFrame([row])

    X_num = row_df[[c for c in num_feats if c in row_df.columns]].copy()

    if cat_feats:
        if ohe is not None:
            X_cat_arr = ohe.transform(row_df[cat_feats].fillna("NA").astype(str))
            cols = []
            for feat, cats in zip(cat_feats, ohe.categories_):
                for val in cats:
                    cols.append(f"{feat}__{val}")
            X_cat = pd.DataFrame(X_cat_arr, columns=cols)
        else:
            X_cat = pd.get_dummies(
                row_df[cat_feats].fillna("NA").astype(str),
                prefix_sep="__",
                columns=cat_feats,
            )
    else:
        X_cat = pd.DataFrame()

    if not X_cat.empty and not X_num.empty:
        X_new = pd.concat(
            [X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1
        )
    elif not X_num.empty:
        X_new = X_num.reset_index(drop=True)
    else:
        X_new = X_cat.reset_index(drop=True)

    num_cols_in_X = X_new.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols_in_X:
        X_new[num_cols_in_X] = scaler.transform(X_new[num_cols_in_X])

    return X_new


# ==============================
# 2. GEMINI UTILITIES (CHATBOT)
# ==============================

@st.cache_resource(show_spinner=False)
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    if not api_key:
        st.warning(
            "⚠️ GEMINI_API_KEY not found. Set it in .streamlit/secrets.toml "
            "or as an environment variable to enable the chatbot."
        )
        return None
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error creating Gemini client: {e}")
        return None


def ask_gemini_chat(client, user_msg: str) -> str:
    if client is None:
        return "Gemini client not available."
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[
                "You are an assistant that answers questions about Electric Vehicles "
                "and EV data analytics. Be clear and concise.",
                f"User question: {user_msg}",
            ],
        )
        return response.text
    except Exception as e:
        return f"Error calling Gemini Chat API: {e}"


# ==============================
# 3. STABILITY AI UTILITIES (IMAGES)
# ==============================

@st.cache_resource(show_spinner=False)
def get_stability_api_key() -> str | None:
    key = st.secrets.get("STABILITY_API_KEY", os.getenv("STABILITY_API_KEY"))
    if not key:
        st.warning(
            "⚠️ STABILITY_API_KEY not found. Create an account on platform.stability.ai, "
            "get an API key, and set it in .streamlit/secrets.toml or env."
        )
        return None
    return key


def generate_ev_images_stability(prompt: str, n: int = 1):
    """Generate EV concept images using Stability AI SD3 endpoint."""
    api_key = get_stability_api_key()
    if api_key is None:
        return [], "Stability API key not available."

    url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
    headers = {
        "authorization": f"Bearer {api_key}",
        "accept": "image/*",
    }

    images = []
    try:
        for _ in range(n):
            data = {
                "prompt": prompt,
                "output_format": "jpeg",
            }
            files = {"none": ""}
            resp = requests.post(url, headers=headers, data=data, files=files)

            if resp.status_code == 200:
                images.append(resp.content)
            else:
                try:
                    msg = resp.json()
                except Exception:
                    msg = resp.text
                return [], f"Stability API error ({resp.status_code}): {msg}"

        if not images:
            return [], "Stability API returned no images."
        return images, None

    except Exception as e:
        return [], f"Error calling Stability API: {e}"


# ==============================
# 4. REINFORCEMENT LEARNING DEMO
# ==============================

def run_pricing_bandit(n_rounds=200, epsilon=0.1, seed=42):
    """Simple epsilon-greedy multi-armed bandit for EV pricing demo."""
    rng = np.random.default_rng(seed)
    true_means = np.array([7.0, 10.0, 8.0])

    n_actions = len(true_means)
    Q = np.zeros(n_actions)
    counts = np.zeros(n_actions)
    rewards = []
    actions = []

    for _ in range(n_rounds):
        if rng.random() < epsilon:
            a = rng.integers(0, n_actions)
        else:
            a = int(np.argmax(Q))

        r = rng.normal(true_means[a], 1.0)
        counts[a] += 1
        Q[a] += (r - Q[a]) / counts[a]

        rewards.append(r)
        actions.append(a)

    cum_rewards = np.cumsum(rewards)
    avg_rewards = cum_rewards / (np.arange(n_rounds) + 1)
    return {
        "true_means": true_means,
        "Q": Q,
        "counts": counts,
        "rewards": np.array(rewards),
        "actions": np.array(actions),
        "avg_rewards": avg_rewards,
    }


# ==============================
# 5. KMEANS CLUSTERING UTILITY
# ==============================

@st.cache_resource(show_spinner=True)
def train_kmeans_model(df: pd.DataFrame, n_clusters: int):
    """Train KMeans clustering on key numeric features."""
    features = []
    for c in [
        "electric_range_capped",
        "electric_range",
        "base_msrp_capped",
        "base_msrp",
        "vehicle_age",
        "price_per_mile",
    ]:
        if c in df.columns:
            features.append(c)

    df_k = df[features].copy()
    df_k = df_k.dropna()
    if df_k.empty:
        return None, None, [], pd.DataFrame()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_k.values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df_cluster = df_k.copy()
    df_cluster["cluster"] = labels
    return kmeans, scaler, features, df_cluster


# ==============================
# 6. MAIN APP
# ==============================
st.title("🚗 EV Cost Predictor & Gen-AI Studio")

df = load_data(DATA_PATH)

# ---- Global realistic ranges from data for sliders ----
# Electric range
if "electric_range_capped" in df.columns:
    r_series = df["electric_range_capped"].dropna()
elif "electric_range" in df.columns:
    r_series = df["electric_range"].dropna()
else:
    r_series = pd.Series([50, 500])

range_min = float(r_series.quantile(0.01))
range_max = float(r_series.quantile(0.99))
range_med = float(r_series.median())

# Price per mile
if "price_per_mile" in df.columns:
    ppm_series = df["price_per_mile"].dropna()
else:
    if "base_msrp_capped" in df.columns:
        base = pd.to_numeric(df["base_msrp_capped"], errors="coerce")
    else:
        base = pd.to_numeric(df["base_msrp"], errors="coerce")
    denom = r_series.reindex(base.index).replace(0, np.nan)
    ppm_series = (base / denom).dropna()

ppm_min = float(ppm_series.quantile(0.01))
ppm_max = float(ppm_series.quantile(0.99))
ppm_med = float(ppm_series.median())

# Base price
if "base_msrp_capped" in df.columns:
    bp_series = pd.to_numeric(df["base_msrp_capped"], errors="coerce").dropna()
else:
    bp_series = pd.to_numeric(df["base_msrp"], errors="coerce").dropna()

base_min = float(bp_series.quantile(0.01))
base_max = float(bp_series.quantile(0.99))
base_med = float(bp_series.median())

# Model year
if "model_year" in df.columns:
    my_series = pd.to_numeric(df["model_year"], errors="coerce").dropna()
    model_year_min = int(my_series.min())
    model_year_max = int(my_series.max())
    model_year_med = int(my_series.median())
else:
    model_year_min, model_year_max, model_year_med = 2010, 2025, 2020

# Train models
(
    price_model,
    scaler_price,
    ohe_price,
    cat_feats_price,
    num_feats_price,
    price_metrics,
    price_feature_names,
) = train_price_regressor(df)

(
    range_model,
    scaler_range,
    ohe_range,
    cat_feats_range,
    num_feats_range,
    range_metrics,
    range_feature_names,
    range_target_col,
) = train_range_regressor(df)

(
    cafv_model,
    scaler_cafv,
    ohe_cafv,
    cat_feats_cafv,
    num_feats_cafv,
    cafv_metrics,
    cafv_feature_names,
) = train_cafv_classifier(df)

tabs = st.tabs(
    [
        "EV Price Predictor",
        "EV Range Forecaster",
        "EV Sales Forecast",
        "EV Eligibility Classification (CAFV)",
        "EV Market Clustering",
        "EV Chatbot (Gemini)",
        "EV Concept Images (Stability AI)",
        "Model Explainability",
        "RL Demo",
        "Help",
    ]
)

# -------- Tab 1: EV Price Predictor --------
with tabs[0]:
    st.header("Predict the Cost of an EV According to Features")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Vehicle Specifications")

        make_list = (
            sorted(df["make"].dropna().unique().tolist())
            if "make" in df.columns
            else []
        )
        ev_type_list = (
            sorted(df["electric_vehicle_type"].dropna().unique().tolist())
            if "electric_vehicle_type" in df.columns
            else []
        )
        range_groups = ["LOW", "MEDIUM", "HIGH", "PREMIUM"]

        make = st.selectbox("Make", make_list) if make_list else st.text_input("Make")
        ev_type = (
            st.selectbox("EV Type", ev_type_list)
            if ev_type_list
            else st.text_input("EV Type")
        )
        range_group = st.selectbox("Range Group", range_groups)

        model_year = st.number_input(
            "Model Year",
            min_value=model_year_min,
            max_value=model_year_max,
            value=model_year_med,
            step=1,
        )

        current_year = 2025
        vehicle_age = current_year - model_year

        electric_range = st.number_input(
            "Electric Range (miles)",
            min_value=float(round(range_min)),
            max_value=float(round(range_max)),
            value=float(round(range_med)),
            step=10.0,
            help=f"Typical values in data are ~{range_min:.0f}–{range_max:.0f} miles.",
        )

        price_per_mile = st.number_input(
            "Expected Price per Mile (₹ or $ / mile)",
            min_value=float(round(ppm_min)),
            max_value=float(round(ppm_max)),
            value=float(round(ppm_med)),
            step=10.0,
            help=f"Typical values in data are ~{ppm_min:.0f}–{ppm_max:.0f} per mile.",
        )

    with colB:
        st.subheader("Model Performance (Price Model)")
        st.metric("R²", f"{price_metrics['R2']:.3f}")
        st.metric("MSE", f"{price_metrics['MSE']:.1f}")
        st.metric("MAE", f"{price_metrics['MAE']:.1f}")
        st.caption("These metrics are computed on a hold-out validation set.")

    st.markdown("---")

    if st.button("🔮 Predict EV Cost"):
        X_new = prepare_single_example(
            df=df,
            ohe=ohe_price,
            cat_feats=cat_feats_price,
            num_feats=num_feats_price,
            scaler=scaler_price,
            make=make,
            ev_type=ev_type,
            range_group=range_group,
            electric_range=electric_range,
            vehicle_age=int(vehicle_age),
            price_per_mile=price_per_mile,
        )
        y_hat = price_model.predict(X_new)[0]
        st.success(f"Predicted Base MSRP (approx.): **{y_hat:,.0f}**")
        st.caption(
            "Note: Approximate value based on Washington EV registration data."
        )


# -------- Tab 2: EV Range Forecaster --------
with tabs[1]:
    st.header("Forecast the Electric Range of an EV")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Vehicle Specifications")

        make_list = (
            sorted(df["make"].dropna().unique().tolist())
            if "make" in df.columns
            else []
        )
        ev_type_list = (
            sorted(df["electric_vehicle_type"].dropna().unique().tolist())
            if "electric_vehicle_type" in df.columns
            else []
        )
        range_groups = ["LOW", "MEDIUM", "HIGH", "PREMIUM"]

        make_r = (
            st.selectbox("Make (Range)", make_list, key="range_make")
            if make_list
            else st.text_input("Make", key="range_make_text")
        )
        ev_type_r = (
            st.selectbox("EV Type (Range)", ev_type_list, key="range_evtype")
            if ev_type_list
            else st.text_input("EV Type", key="range_evtype_text")
        )
        range_group_r = st.selectbox(
            "Range Group (expected)", range_groups, key="range_group"
        )

        model_year_r = st.number_input(
            "Model Year",
            min_value=model_year_min,
            max_value=model_year_max,
            value=model_year_med,
            step=1,
            key="range_model_year",
        )
        current_year = 2025
        vehicle_age_r = current_year - model_year_r

        base_price_guess = st.number_input(
            "Estimated base price (₹ or $)",
            min_value=float(round(base_min, -3)),
            max_value=float(round(base_max, -3)),
            value=float(round(base_med, -3)),
            step=1000.0,
            key="range_base_price",
            help=f"Typical prices in data are ~{base_min:,.0f}–{base_max:,.0f}.",
        )

        price_per_mile_r = st.number_input(
            "Estimated price per mile (₹ or $ / mile)",
            min_value=float(round(ppm_min)),
            max_value=float(round(ppm_max)),
            value=float(round(ppm_med)),
            step=10.0,
            key="range_ppm",
            help=f"Typical values in data are ~{ppm_min:.0f}–{ppm_max:.0f} per mile.",
        )

        electric_range_placeholder = base_price_guess / max(price_per_mile_r, 1.0)

    with colB:
        st.subheader("Model Performance (Range Model)")
        st.metric("R²", f"{range_metrics['R2']:.3f}")
        st.metric("MSE", f"{range_metrics['MSE']:.1f}")
        st.metric("MAE", f"{range_metrics['MAE']:.1f}")
        st.caption(f"Target variable: **{range_target_col}**")

    st.markdown("---")

    if st.button("📈 Forecast EV Range"):
        X_new_r = prepare_single_example(
            df=df,
            ohe=ohe_range,
            cat_feats=cat_feats_range,
            num_feats=num_feats_range,
            scaler=scaler_range,
            make=make_r,
            ev_type=ev_type_r,
            range_group=range_group_r,
            electric_range=electric_range_placeholder,
            vehicle_age=int(vehicle_age_r),
            price_per_mile=price_per_mile_r,
        )
        range_hat = range_model.predict(X_new_r)[0]
        st.success(f"Predicted Electric Range (approx.): **{range_hat:.1f} miles**")
        st.caption(
            "Note: This is a statistical forecast based on similar vehicles in the dataset."
        )


# -------- Tab 3: EV Sales Forecast --------
with tabs[2]:
    st.header("Predict / Forecast EV Sales (Registrations)")

    st.write(
        "We approximate **EV sales** using the number of registered vehicles "
        "per year in the Washington dataset, and fit a simple Linear Regression "
        "model to forecast future sales."
    )

    make_list_all = ["ALL MAKES"]
    if "make" in df.columns:
        make_list_all += sorted(df["make"].dropna().unique().tolist())
    chosen_make = st.selectbox("Choose Make for Sales Forecast", make_list_all)

    df_sales = df.copy()
    df_sales = df_sales.dropna(subset=["model_year"])
    if chosen_make != "ALL MAKES":
        df_sales = df_sales[df_sales["make"] == chosen_make]

    yearly_counts = df_sales.groupby("model_year")["vin_1_10"].count().reset_index()
    yearly_counts = yearly_counts.sort_values("model_year")
    yearly_counts.columns = ["year", "sales"]

    if yearly_counts.empty:
        st.warning("No data available for this make.")
    else:
        st.subheader("Historical Registrations (Proxy for Sales)")
        st.dataframe(yearly_counts, width="stretch")

        X_year = yearly_counts[["year"]].values
        y_sales = yearly_counts["sales"].values
        lr = LinearRegression()
        lr.fit(X_year, y_sales)

        last_year = int(yearly_counts["year"].max())
        future_year = st.slider(
            "Forecast up to year",
            min_value=last_year + 1,
            max_value=last_year + 10,
            value=last_year + 3,
        )

        all_years = np.arange(yearly_counts["year"].min(), future_year + 1)
        preds = lr.predict(all_years.reshape(-1, 1))

        fig, ax = plt.subplots()
        ax.plot(yearly_counts["year"], yearly_counts["sales"], "o-", label="Historical")
        ax.plot(all_years, preds, "--", label="Linear forecast")
        ax.set_xlabel("Year")
        ax.set_ylabel("Registrations (proxy for sales)")
        title_make = chosen_make if chosen_make != "ALL MAKES" else "All makes"
        ax.set_title(f"EV Sales Trend – {title_make}")
        ax.legend()
        st.pyplot(fig)

        forecast_val = float(preds[-1])
        st.success(
            f"Forecasted registrations (sales proxy) in **{future_year}** "
            f"for **{title_make}** ≈ **{forecast_val:.0f} vehicles**"
        )


# -------- Tab 4: EV Eligibility Classification (CAFV) --------
with tabs[3]:
    st.header("EV Eligibility Classification (CAFV)")

    st.write(
        """
        This tab performs **supervised classification** to predict whether an EV
        is **CAFV Eligible** or **Not Eligible** using a RandomForestClassifier.

        - 1 → "Clean Alternative Fuel Vehicle Eligible"  
        - 0 → all other statuses
        """
    )

    if cafv_model is None:
        st.error("CAFV model could not be trained (missing column).")
    else:
        colA, colB = st.columns(2)

        with colA:
            make_list = (
                sorted(df["make"].dropna().unique().tolist())
                if "make" in df.columns
                else []
            )
            ev_type_list = (
                sorted(df["electric_vehicle_type"].dropna().unique().tolist())
                if "electric_vehicle_type" in df.columns
                else []
            )
            range_groups = ["LOW", "MEDIUM", "HIGH", "PREMIUM"]

            make_c = (
                st.selectbox("Make (Classification)", make_list, key="cafv_make")
                if make_list
                else st.text_input("Make", key="cafv_make_text")
            )
            ev_type_c = (
                st.selectbox("EV Type (Classification)", ev_type_list, key="cafv_evtype")
                if ev_type_list
                else st.text_input("EV Type", key="cafv_evtype_text")
            )
            range_group_c = st.selectbox(
                "Range Group", range_groups, key="cafv_range_group"
            )

            model_year_c = st.number_input(
                "Model Year",
                min_value=model_year_min,
                max_value=model_year_max,
                value=model_year_med,
                step=1,
                key="cafv_model_year",
            )
            current_year = 2025
            vehicle_age_c = current_year - model_year_c

            electric_range_c = st.number_input(
                "Estimated Electric Range (miles)",
                min_value=float(round(range_min)),
                max_value=float(round(range_max)),
                value=float(round(range_med)),
                step=10.0,
                key="cafv_range",
                help=f"Typical values in data are ~{range_min:.0f}–{range_max:.0f} miles.",
            )

            price_per_mile_c = st.number_input(
                "Estimated Price per Mile (₹ or $ / mile)",
                min_value=float(round(ppm_min)),
                max_value=float(round(ppm_max)),
                value=float(round(ppm_med)),
                step=10.0,
                key="cafv_ppm",
                help=f"Typical values in data are ~{ppm_min:.0f}–{ppm_max:.0f} per mile.",
            )

        with colB:
            st.subheader("Validation Metrics (Binary Classification)")
            st.metric("Accuracy", f"{cafv_metrics.get('accuracy', 0):.3f}")
            st.metric("Precision", f"{cafv_metrics.get('precision', 0):.3f}")
            st.metric("Recall", f"{cafv_metrics.get('recall', 0):.3f}")
            st.metric("F1-score", f"{cafv_metrics.get('f1', 0):.3f}")

            cm = cafv_metrics.get("confusion_matrix")
            if cm is not None and cm.size == 4:
                st.caption("Confusion Matrix (rows=true, cols=pred, order=[0,1])")
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual: Not Eligible (0)", "Actual: Eligible (1)"],
                    columns=["Pred: Not Eligible (0)", "Pred: Eligible (1)"],
                )
                st.dataframe(cm_df, width="stretch")

        st.markdown("---")

        if st.button("✅ Predict CAFV Eligibility"):
            X_new_c = prepare_single_example(
                df=df,
                ohe=ohe_cafv,
                cat_feats=cat_feats_cafv,
                num_feats=num_feats_cafv,
                scaler=scaler_cafv,
                make=make_c,
                ev_type=ev_type_c,
                range_group=range_group_c,
                electric_range=electric_range_c,
                vehicle_age=int(vehicle_age_c),
                price_per_mile=price_per_mile_c,
            )
            y_hat_c = cafv_model.predict(X_new_c)[0]
            prob = cafv_model.predict_proba(X_new_c)[0, 1]

            label = "Clean Alternative Fuel Vehicle Eligible" if y_hat_c == 1 else "Not Eligible"
            st.success(
                f"Predicted CAFV status: **{label}**  \n"
                f"Model confidence (P[Eligible]): **{prob:.2f}**"
            )
            st.caption(
                "Note: This is a predictive estimate. Actual eligibility rules are determined by policy."
            )


# -------- Tab 5: EV Market Clustering --------
with tabs[4]:
    st.header("EV Market Clustering (Unsupervised – KMeans)")

    st.write(
        """
        This tab demonstrates **unsupervised learning** using the K-Means
        clustering algorithm. We cluster EVs using numerical features like
        range, price and age to discover hidden market segments.
        """
    )

    n_clusters = st.slider("Number of clusters (k)", 3, 6, 4)

    kmeans, scaler_km, used_features, df_cluster = train_kmeans_model(df, n_clusters)

    if df_cluster.empty:
        st.warning("Not enough numeric data to perform clustering.")
    else:
        st.subheader("Clustered Data (sample)")
        st.dataframe(df_cluster.head(20), width="stretch")

        x_feat = (
            "electric_range_capped"
            if "electric_range_capped" in df_cluster.columns
            else "electric_range"
        )
        y_feat = (
            "price_per_mile"
            if "price_per_mile" in df_cluster.columns
            else (
                "base_msrp_capped"
                if "base_msrp_capped" in df_cluster.columns
                else "base_msrp"
            )
        )

        if x_feat in df_cluster.columns and y_feat in df_cluster.columns:
            fig, ax = plt.subplots()
            scatter = ax.scatter(
                df_cluster[x_feat],
                df_cluster[y_feat],
                c=df_cluster["cluster"],
                cmap="tab10",
                alpha=0.7,
            )
            ax.set_xlabel(x_feat)
            ax.set_ylabel(y_feat)
            ax.set_title("KMeans clusters in EV feature space")
            legend1 = ax.legend(
                *scatter.legend_elements(), title="Cluster", loc="best"
            )
            ax.add_artist(legend1)
            st.pyplot(fig)

        st.subheader("Cluster Summary (mean feature values)")
        cluster_summary = df_cluster.groupby("cluster")[used_features].mean()
        st.dataframe(cluster_summary, width="stretch")

        st.markdown(
            """
            **How to interpret:**

            - Each cluster groups EVs with similar range, price and age.  
            - You can label clusters in your report, e.g.:  
              - Cluster 0 → Low-cost, low-range commuters  
              - Cluster 1 → Premium, long-range EVs  
              - Cluster 2 → Mid-range value EVs  
            """
        )


# -------- Tab 6: EV Chatbot (Gemini) --------
with tabs[5]:
    st.header("EV Gen-AI Chatbot (Gemini)")

    st.write(
        "Ask anything about Electric Vehicles, EV range, pricing, charging, "
        "or your ML project. This uses Google's Gemini text model."
    )

    gemini_client = get_gemini_client()

    user_msg = st.text_area(
        "Your question:",
        height=150,
        placeholder='Example: "Explain the main factors that affect EV cost and range."',
    )

    if st.button("Ask Chatbot"):
        if not user_msg.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Thinking..."):
                reply = ask_gemini_chat(gemini_client, user_msg.strip())
            st.markdown("**Chatbot:**")
            st.write(reply)


# -------- Tab 7: EV Concept Images (Stability AI) --------
with tabs[6]:
    st.header("EV Concept Images (Stability AI – Stable Diffusion 3.5)")

    st.write(
        "Describe an Electric Vehicle concept and generate AI images using "
        "Stability AI's Stable Diffusion 3.5 image API."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        base_prompt = st.text_area(
            "Describe your EV concept:",
            height=140,
            placeholder=(
                "Example: \"A futuristic electric SUV in a neon-lit city at night, "
                "top view, cinematic lighting\""
            ),
        )

    with col2:
        style = st.selectbox(
            "Style",
            ["Realistic", "Futuristic concept art", "Cartoon", "Minimal flat illustration"],
            index=1,
        )
        color_theme = st.selectbox(
            "Color theme",
            ["Any", "Neon cyberpunk", "Pastel", "Monochrome"],
            index=0,
        )
        n_images = st.slider("Number of images", 1, 4, 2)

    if st.button("🎨 Generate EV Images"):
        if not base_prompt.strip():
            st.warning("Please type a description first.")
        else:
            full_prompt = (
                f"{base_prompt.strip()} | style: {style}, "
                f"color theme: {color_theme}, electric vehicle concept art"
            )
            with st.spinner("Calling Stability AI..."):
                imgs, err = generate_ev_images_stability(full_prompt, n=n_images)

            if err:
                st.error(err)
            else:
                st.success(f"Generated {len(imgs)} image(s) using Stability AI.")
                cols = st.columns(len(imgs))
                for i, img_bytes in enumerate(imgs):
                    with cols[i]:
                        st.image(
                            img_bytes,
                            caption=f"EV concept #{i+1}",
                            width="stretch",
                        )


# -------- Tab 8: Model Explainability --------
with tabs[7]:
    st.header("Model Explainability – Feature Importance (Price Model)")

    st.write(
        "This section explains **which features matter most** for the RandomForest "
        "EV cost prediction model."
    )

    importances = price_model.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": price_feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    top_n = st.slider("Number of top features to show", 5, min(30, len(fi_df)), 15)
    top_fi = fi_df.head(top_n).set_index("feature")

    st.subheader("Top feature importances (Price model)")
    st.bar_chart(top_fi)

    st.subheader("Raw importance table")
    st.dataframe(fi_df.head(50), width="stretch")

    st.markdown(
        """
        - Higher importance = stronger impact on predicted EV price.  
        - E.g., if `electric_range_capped` and `vehicle_age` are at the top,
          the model relies heavily on range and age to estimate cost.  
        """
    )


# -------- Tab 9: RL Demo --------
with tabs[8]:
    st.header("Reinforcement Learning Demo – Dynamic Pricing Bandit")

    st.write(
        """
        This tab demonstrates **Reinforcement Learning (RL)** using a simple
        **multi-armed bandit** problem for EV pricing:

        - The agent chooses between 3 price levels: **Low, Medium, High**.  
        - Each price level has a hidden expected profit (reward).  
        - Using an **ε-greedy policy**, the agent explores different prices and 
          gradually learns which price gives higher long-term profit.
        """
    )

    n_rounds = st.slider("Number of interaction steps", 50, 500, 200, step=50)
    epsilon = st.slider("Exploration rate ε", 0.01, 0.5, 0.1)

    if st.button("▶ Run RL Simulation"):
        res = run_pricing_bandit(n_rounds=n_rounds, epsilon=epsilon)

        st.subheader("Learned value estimates Q(a)")
        price_labels = ["Low price", "Medium price", "High price"]
        df_q = pd.DataFrame(
            {
                "Price level": price_labels,
                "True mean profit": res["true_means"],
                "Learned Q(a)": res["Q"],
                "Times chosen": res["counts"],
            }
        )
        st.dataframe(df_q, width="stretch")

        fig, ax = plt.subplots()
        ax.plot(res["avg_rewards"])
        ax.set_xlabel("Step")
        ax.set_ylabel("Average reward")
        ax.set_title("Learning curve of ε-greedy pricing agent")
        st.pyplot(fig)

        st.markdown(
            """
            - Early steps: agent explores randomly → average reward is noisy.  
            - Later: it learns which price is best and chooses it more often →  
              average reward increases and stabilizes.  
            - This shows how RL can optimize **EV pricing strategies** over time.
            """
        )


# -------- Tab 10: Help / Instructions --------
with tabs[9]:
    st.header("Project Help & Instructions")

    st.markdown(
        """
        ### Mapping to Workshop Requirements

        1. **Regression (Supervised Learning)**  
           - *EV Price Predictor* – RandomForestRegressor → Base MSRP  
           - *EV Range Forecaster* – RandomForestRegressor → Electric range  

        2. **Classification (Supervised Learning)**  
           - *EV Eligibility Classification (CAFV)* – RandomForestClassifier  
             - Binary label (Eligible / Not Eligible)  
             - Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix  

        3. **Unsupervised Learning (Clustering)**  
           - *EV Market Clustering* – KMeans (k = 3–6)  
             - Clusters based on range, price, age, price_per_mile  

        4. **Gen AI**  
           - *EV Chatbot (Gemini)* – text-based QA about EVs  
           - *EV Concept Images (Stability AI)* – EV concept art generation  

        5. **Reinforcement Learning**  
           - *RL Demo* – ε-greedy multi-armed bandit for dynamic EV pricing  

        6. **Evaluation Metrics**  
           - Regression: MSE, MAE, R²  
           - Classification: Accuracy, Precision, Recall, F1  
           - Clustering: qualitative interpretation of cluster centers  

        7. **Streamlit Frontend**  
           - All models are integrated in an interactive multi-tab web app.  

        ### How to run

        ```bash
        pip install streamlit scikit-learn pandas numpy google-genai requests matplotlib
        streamlit run app.py
        ```

        Configure keys in `.streamlit/secrets.toml`:

        ```toml
        GEMINI_API_KEY = "your_gemini_key_here"
        STABILITY_API_KEY = "your_stability_key_here"
        ```
        """
    )
