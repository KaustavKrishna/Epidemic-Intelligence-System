"""
============================================================
  EPIDEMIC INTELLIGENCE DASHBOARD — ML Pipeline
  Hackathon Project | COVID-19 Outbreak Risk Predictor
============================================================

PIPELINE OVERVIEW:
  1. Data Ingestion     → Pull JHU COVID-19 time-series CSV
  2. Feature Engineering → Growth rate, 7-day MA, risk ratios
  3. Model Training     → XGBoost + Random Forest
  4. Risk Scoring       → Low / Medium / High per country
  5. Early Warning      → Spike detection algorithm
  6. Visualization      → Interactive Plotly dashboard
  7. Export             → JSON summary for the web frontend

DEPENDENCIES:
  pip install pandas numpy scikit-learn xgboost plotly requests
============================================================
"""

# ─────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import requests
import warnings
import json
from datetime import datetime, timedelta

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# SECTION 2: DATA INGESTION
# ─────────────────────────────────────────────

# Johns Hopkins GitHub URL (confirmed cases)
JHU_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)


def load_jhu_data(url: str = JHU_URL) -> pd.DataFrame:
    """
    Load and melt the JHU wide-format CSV into a long-format DataFrame.

    Raw format: Each column after 'Long' is a date with cumulative cases.
    Output format: [Country, Date, Confirmed] — one row per country per day.
    """
    print("📥 Fetching JHU COVID-19 data...")
    df = pd.read_csv(url)

    # Drop sub-region (Province/State) and coordinate columns
    df = df.drop(columns=["Province/State", "Lat", "Long"], errors="ignore")

    # Aggregate to country level (sum all provinces)
    df = df.groupby("Country/Region", as_index=False).sum()

    # Melt from wide to long format
    date_columns = [c for c in df.columns if c != "Country/Region"]
    df = df.melt(
        id_vars="Country/Region",
        value_vars=date_columns,
        var_name="Date",
        value_name="Confirmed",
    )

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"])
    df.rename(columns={"Country/Region": "Country"}, inplace=True)
    df.sort_values(["Country", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"✅ Loaded {len(df['Country'].unique())} countries, {len(df)} rows total.")
    return df


# ─────────────────────────────────────────────
# SECTION 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create predictive features from raw cumulative case counts.

    Features created:
    ┌─────────────────────────────────────────────────────────────────┐
    │ daily_new_cases   → Day-over-day increase in confirmed cases     │
    │ growth_rate       → % change in new cases vs previous day       │
    │ ma_7              → 7-day moving average of daily new cases      │
    │ ma_14             → 14-day moving average (trend smoother)       │
    │ doubling_time     → Estimated days to double at current growth   │
    │ case_acceleration → Rate of change of growth (2nd derivative)   │
    │ lag_1 / lag_7     → Lagged case counts (autoregression signals) │
    └─────────────────────────────────────────────────────────────────┘
    """
    print("🔧 Engineering features...")

    # Work per country to avoid cross-country contamination
    groups = []
    for country, grp in df.groupby("Country"):
        grp = grp.copy().sort_values("Date")

        # Daily new cases (diff of cumulative)
        grp["daily_new_cases"] = grp["Confirmed"].diff().clip(lower=0)

        # 7-day and 14-day moving averages
        grp["ma_7"]  = grp["daily_new_cases"].rolling(7,  min_periods=1).mean()
        grp["ma_14"] = grp["daily_new_cases"].rolling(14, min_periods=1).mean()

        # Growth rate: % change in 7-day MA vs 7 days ago
        grp["growth_rate"] = grp["ma_7"].pct_change(periods=7) * 100

        # Case acceleration (growth of growth)
        grp["case_acceleration"] = grp["growth_rate"].diff()

        # Doubling time: ln(2) / ln(1 + growth_rate/100)
        # Clip growth_rate to avoid log(0) or negative
        safe_gr = grp["growth_rate"].clip(lower=0.1) / 100
        grp["doubling_time"] = np.log(2) / np.log(1 + safe_gr)
        grp["doubling_time"] = grp["doubling_time"].replace([np.inf, -np.inf], np.nan)

        # Autoregression lags
        grp["lag_1"] = grp["daily_new_cases"].shift(1)
        grp["lag_7"] = grp["daily_new_cases"].shift(7)

        # Day-of-week (captures weekly reporting patterns)
        grp["day_of_week"] = grp["Date"].dt.dayofweek

        groups.append(grp)

    result = pd.concat(groups, ignore_index=True)
    result.dropna(subset=["ma_7", "lag_7"], inplace=True)  # Drop early rows with NaN lags
    print(f"✅ Feature engineering complete. Dataset shape: {result.shape}")
    return result


# ─────────────────────────────────────────────
# SECTION 4: RISK SCORING ENGINE
# ─────────────────────────────────────────────

def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a composite Outbreak Risk Score (0–100) to each country/date.

    Formula (weighted sum, normalized):
        risk = 0.40 × growth_rate_norm
              + 0.30 × ma_7_norm
              + 0.20 × acceleration_norm
              + 0.10 × recency_factor

    Labels:
        0–33  → 🟢 Low Risk
        34–66 → 🟡 Medium Risk
        67–100 → 🔴 High Risk
    """
    print("🔥 Computing outbreak risk scores...")

    def normalize(series):
        """Min-max normalization to [0, 1], handles NaN gracefully."""
        mn, mx = series.min(), series.max()
        if mx == mn:
            return series * 0
        return (series - mn) / (mx - mn)

    df = df.copy()

    # Normalize each component
    gr_norm   = normalize(df["growth_rate"].clip(lower=0))
    ma_norm   = normalize(df["ma_7"])
    acc_norm  = normalize(df["case_acceleration"].clip(lower=0))

    # Recency factor: most recent 30 days get a slight boost
    max_date = df["Date"].max()
    days_ago = (max_date - df["Date"]).dt.days
    recency  = normalize(1 / (days_ago + 1))  # Higher for recent dates

    # Composite risk score (0–100)
    df["risk_score"] = (
        0.40 * gr_norm +
        0.30 * ma_norm +
        0.20 * acc_norm +
        0.10 * recency
    ) * 100

    # Risk label
    df["risk_label"] = pd.cut(
        df["risk_score"],
        bins=[-1, 33, 66, 101],
        labels=["Low", "Medium", "High"],
    )

    print("✅ Risk scores assigned.")
    return df


# ─────────────────────────────────────────────
# SECTION 5: EARLY WARNING SYSTEM
# ─────────────────────────────────────────────

def detect_spikes(df: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    """
    Early Warning System: flag countries with abnormal case spikes.

    Method: Z-score on 7-day moving average.
    If z-score > z_threshold → spike detected (potential outbreak start).

    A z-score measures how many standard deviations above normal a value is.
    z_threshold=2.5 means "flag if 2.5× above the recent trend".
    """
    print(f"⚠️  Running Early Warning spike detection (z={z_threshold})...")

    alerts = []
    for country, grp in df.groupby("Country"):
        grp = grp.copy().sort_values("Date")
        mean_ma = grp["ma_7"].mean()
        std_ma  = grp["ma_7"].std()

        if std_ma == 0 or np.isnan(std_ma):
            continue

        grp["z_score"] = (grp["ma_7"] - mean_ma) / std_ma
        spikes = grp[grp["z_score"] > z_threshold]

        for _, row in spikes.iterrows():
            alerts.append({
                "Country":    country,
                "Date":       row["Date"].strftime("%Y-%m-%d"),
                "MA_7":       round(row["ma_7"], 1),
                "Z_Score":    round(row["z_score"], 2),
                "Risk_Label": str(row.get("risk_label", "Unknown")),
            })

    alert_df = pd.DataFrame(alerts)
    print(f"✅ {len(alert_df)} spike events detected across all countries.")
    return alert_df


# ─────────────────────────────────────────────
# SECTION 6: ML MODEL TRAINING
# ─────────────────────────────────────────────

FEATURE_COLS = ["lag_1", "lag_7", "ma_7", "ma_14", "growth_rate",
                "case_acceleration", "doubling_time", "day_of_week"]
TARGET_COL   = "daily_new_cases"


def prepare_ml_data(df: pd.DataFrame):
    """
    Prepare clean feature matrix X and target vector y for ML training.
    Drops rows with NaN in any feature column.
    """
    subset = df[FEATURE_COLS + [TARGET_COL]].dropna()
    X = subset[FEATURE_COLS]
    y = subset[TARGET_COL]
    return X, y


def train_models(X_train, y_train):
    """
    Train three models:
      1. Linear Regression   — simple baseline
      2. Random Forest       — ensemble, handles non-linearity
      3. XGBoost             — gradient boosting, best performer

    Returns a dict of {model_name: trained_model}.
    """
    print("🤖 Training ML models...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {}

    # ── Model 1: Linear Regression (Baseline) ──
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models["Linear Regression"] = (lr, scaler, True)   # True = needs scaling
    print("   ✓ Linear Regression trained")

    # ── Model 2: Random Forest ──
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = (rf, None, False)         # False = no scaling needed
    print("   ✓ Random Forest trained")

    # ── Model 3: XGBoost ──
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    models["XGBoost"] = (xgb_model, None, False)
    print("   ✓ XGBoost trained")

    return models


def evaluate_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models and print a comparison table.
    Metrics: MAE, RMSE, R²
    """
    print("\n📊 Model Evaluation Results:")
    print("─" * 55)
    print(f"{'Model':<22} {'MAE':>8} {'RMSE':>10} {'R²':>8}")
    print("─" * 55)

    results = []
    for name, (model, scaler, needs_scale) in models.items():
        X_eval = scaler.transform(X_test) if needs_scale else X_test
        preds  = model.predict(X_eval).clip(min=0)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

        print(f"{name:<22} {mae:>8,.0f} {rmse:>10,.0f} {r2:>8.4f}")
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    print("─" * 55)
    return pd.DataFrame(results)


def get_feature_importance(xgb_model, feature_names: list) -> pd.DataFrame:
    """Extract and sort feature importances from the XGBoost model."""
    importances = xgb_model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    return fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# SECTION 7: VISUALIZATION
# ─────────────────────────────────────────────

def plot_dashboard(df: pd.DataFrame, country: str = "US",
                   models: dict = None, X_test=None, y_test=None):
    """
    Build an interactive 4-panel Plotly dashboard:
      Panel 1: Predicted vs Actual cases (top model)
      Panel 2: 7-day MA trend + risk bands
      Panel 3: Risk score over time
      Panel 4: Feature importance bar chart
    """
    print(f"\n📈 Building dashboard for: {country}")
    cdf = df[df["Country"] == country].sort_values("Date").copy()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "📊 Predicted vs Actual Daily Cases",
            "📉 7-Day MA Trend + Risk Bands",
            "🔥 Outbreak Risk Score Over Time",
            "🧠 Feature Importance (XGBoost)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # ── Panel 1: Predicted vs Actual ──
    if models and X_test is not None:
        xgb_model, _, _ = models["XGBoost"]
        preds = xgb_model.predict(X_test).clip(min=0)
        fig.add_trace(go.Scatter(y=y_test.values[:200], name="Actual",
                                 line=dict(color="#00d4ff", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(y=preds[:200], name="Predicted",
                                 line=dict(color="#ff6b35", width=2, dash="dot")), row=1, col=1)

    # ── Panel 2: MA Trend ──
    fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["ma_7"], name="7-Day MA",
                             fill="tozeroy", fillcolor="rgba(0,212,255,0.1)",
                             line=dict(color="#00d4ff", width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["ma_14"], name="14-Day MA",
                             line=dict(color="#ffdd57", width=1.5, dash="dash")), row=1, col=2)

    # ── Panel 3: Risk Score ──
    if "risk_score" in cdf.columns:
        colors = cdf["risk_label"].map({"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"})
        fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["risk_score"],
                                 mode="lines", name="Risk Score",
                                 line=dict(color="#ff6b35", width=2)), row=2, col=1)

    # ── Panel 4: Feature Importance ──
    if models:
        xgb_model, _, _ = models["XGBoost"]
        fi_df = get_feature_importance(xgb_model, FEATURE_COLS)
        fig.add_trace(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"],
                             orientation="h", marker_color="#00d4ff",
                             name="Importance"), row=2, col=2)

    fig.update_layout(
        title=f"Epidemic Intelligence Dashboard — {country}",
        template="plotly_dark",
        height=750,
        showlegend=True,
        font=dict(family="monospace", size=11),
    )
    fig.write_html("epidemic_dashboard.html")
    print("✅ Dashboard saved to epidemic_dashboard.html")
    return fig


# ─────────────────────────────────────────────
# SECTION 8: EXPORT SUMMARY FOR WEB FRONTEND
# ─────────────────────────────────────────────

def export_summary(df: pd.DataFrame, alert_df: pd.DataFrame, output_path: str = "summary.json"):
    """
    Export a JSON snapshot of the latest risk status for each country.
    Used by the web frontend to display live risk data.
    """
    latest = df.sort_values("Date").groupby("Country").last().reset_index()
    latest = latest[["Country", "Confirmed", "daily_new_cases", "ma_7",
                     "growth_rate", "risk_score", "risk_label"]]
    latest = latest.dropna(subset=["risk_score"])
    latest["risk_label"] = latest["risk_label"].astype(str)

    # Top 10 high-risk countries
    top10 = latest.nlargest(10, "risk_score")[
        ["Country", "risk_score", "risk_label", "ma_7", "growth_rate"]
    ].round(2).to_dict(orient="records")

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_countries": int(latest.shape[0]),
        "high_risk_count":   int((latest["risk_label"] == "High").sum()),
        "medium_risk_count": int((latest["risk_label"] == "Medium").sum()),
        "low_risk_count":    int((latest["risk_label"] == "Low").sum()),
        "top_10_high_risk":  top10,
        "total_alerts":      int(len(alert_df)),
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"✅ Summary exported to {output_path}")
    return summary


# ─────────────────────────────────────────────
# SECTION 9: MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🦠 EPIDEMIC INTELLIGENCE DASHBOARD — ML PIPELINE")
    print("=" * 60)

    # Step 1: Load Data
    df_raw = load_jhu_data()

    # Step 2: Feature Engineering
    df = engineer_features(df_raw)

    # Step 3: Risk Scoring
    df = compute_risk_score(df)

    # Step 4: Early Warning Detection
    alert_df = detect_spikes(df, z_threshold=2.5)
    print("\nTop 5 Spike Alerts:")
    print(alert_df.head())

    # Step 5: ML Training
    X, y = prepare_ml_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # time-aware split
    )
    models = train_models(X_train, y_train)

    # Step 6: Evaluation
    eval_df = evaluate_models(models, X_test, y_test)

    # Step 7: Feature Importance
    xgb_model, _, _ = models["XGBoost"]
    fi_df = get_feature_importance(xgb_model, FEATURE_COLS)
    print("\n🧠 Top Feature Importances:")
    print(fi_df.to_string(index=False))

    # Step 8: Dashboard
    plot_dashboard(df, country="US", models=models, X_test=X_test, y_test=y_test)

    # Step 9: Export JSON for website
    summary = export_summary(df, alert_df)
    print(f"\n🌍 Risk Summary: {summary['high_risk_count']} High | "
          f"{summary['medium_risk_count']} Medium | {summary['low_risk_count']} Low")

    print("\n✅ Pipeline complete! Files generated:")
    print("   • epidemic_dashboard.html  → Interactive Plotly dashboard")
    print("   • summary.json            → Risk data for web frontend")


if __name__ == "__main__":
    main()
