import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_MART_PATH = PROJECT_ROOT / "data/features/feature_mart.csv"
PBP_PATH = PROJECT_ROOT / "data/processed/pbp_with_possessions.csv"

FATIGUE_MODEL_PATH = PROJECT_ROOT / "data/models/fatigue_model.pkl"
CLUTCH_MODEL_PATH = PROJECT_ROOT / "data/models/clutch_classifier.pkl"

KEYS = ["game_id", "playerNameI", "period"]

# -----------------------------
# Utilities
# -----------------------------
def safe_read_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None

def load_model(path: Path):
    return joblib.load(path) if path.exists() else None

def percentile_rank(series: pd.Series, value: float) -> float:
    if series is None or series.empty:
        return float("nan")
    return float((series <= value).mean() * 100.0)

def format_level(p: float) -> str:
    if np.isnan(p):
        return "UNKNOWN"
    if p >= 90:
        return "VERY HIGH"
    if p >= 75:
        return "HIGH"
    if p >= 50:
        return "MEDIUM"
    if p >= 25:
        return "LOW"
    return "VERY LOW"

def align_to_model(X: pd.DataFrame, model):
    if model is None:
        return X
    if hasattr(model, "feature_names_in_"):
        return X.reindex(columns=model.feature_names_in_, fill_value=0)
    return X

def find_any_col(df, keywords):
    # returns first matching column name from df
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in keywords):
            return c
    return None

# -----------------------------
# Load training distributions (for calibration + percentiles)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_reference_tables():
    mart = safe_read_csv(FEATURE_MART_PATH)
    pbp = safe_read_csv(PBP_PATH)
    return mart, pbp

@st.cache_resource(show_spinner=False)
def load_models():
    fatigue_model = load_model(FATIGUE_MODEL_PATH)
    clutch_model = load_model(CLUTCH_MODEL_PATH)
    return fatigue_model, clutch_model

mart, pbp = load_reference_tables()
fatigue_model, clutch_model = load_models()

st.set_page_config(page_title="CDIS — Scenario Decision Dashboard", layout="wide")
st.title("🏀 CDIS — Scenario Decision Dashboard (No game_id required)")
st.caption("Enter the current game situation → get fatigue, clutch, momentum, and lineup stability insights + decision recommendations.")

if mart is None:
    st.error("Missing data/features/feature_mart.csv. Build Day 14 feature mart first.")
    st.stop()

# -----------------------------
# Sidebar: scenario inputs
# -----------------------------
st.sidebar.header("🎮 Current Game Situation")

period = st.sidebar.selectbox("Period", [1, 2, 3, 4], index=3)
clock_sec = st.sidebar.number_input("Clock (seconds remaining in period)", 0, 720, 180)
score_home = st.sidebar.number_input("Home Score", 0, 200, 100)
score_away = st.sidebar.number_input("Away Score", 0, 200, 98)
score_diff = abs(score_home - score_away)

st.sidebar.divider()
st.sidebar.subheader("Recent Window (last ~3–5 possessions)")

# Inputs you can realistically know in-game
recent_points_for = st.sidebar.number_input("Points scored by your team (recent window)", 0, 30, 8)
recent_points_against = st.sidebar.number_input("Points scored by opponent (recent window)", 0, 30, 0)
recent_shot_attempts = st.sidebar.number_input("Your team's shot attempts (recent window)", 0, 30, 6)
recent_turnovers = st.sidebar.number_input("Your team's turnovers (recent window)", 0, 10, 1)

st.sidebar.divider()
st.sidebar.subheader("Player Load Inputs (for the player you’re deciding on)")

minutes_proxy = st.sidebar.slider("Minutes played proxy in current period", 0.0, 12.0, 6.0, 0.25)
high_intensity_events = st.sidebar.number_input("High intensity events (closeouts/contests/sprints proxy)", 0, 50, 12)
continuous_possessions = st.sidebar.number_input("Continuous possessions without rest", 0, 20, 4)
usage_burst = st.sidebar.slider("Usage burst (0–1)", 0.0, 1.0, 0.55, 0.01)
defensive_events = st.sidebar.number_input("Defensive events (steals/blocks/def rebounds proxy)", 0, 25, 5)

st.sidebar.divider()
st.sidebar.subheader("Rotation / Lineup")

lineup_changes_recent = st.sidebar.number_input("Lineup changes in recent window", 0, 10, 2)
stable_core = st.sidebar.checkbox("Stable 3–4 man core on court?", value=True)

st.sidebar.divider()
mode = st.sidebar.radio("Decision Type", ["Sub/Rest decision", "Offensive usage", "Clutch shot selection", "General"])
what_if_rest = st.sidebar.checkbox("What-if rest simulation (reduce load 20%)", value=True)

# -----------------------------
# Construct features (scenario → engineered proxies)
# -----------------------------
# Momentum proxy
run_margin = recent_points_for - recent_points_against
run_length = recent_points_for if recent_points_against == 0 else 0
momentum_score = (run_margin / 10.0) + (run_length / 12.0) - (recent_turnovers / 5.0)

# Lineup stability proxy
lineup_stability_proxy = 1.0 / (1.0 + lineup_changes_recent)
if stable_core:
    lineup_stability_proxy += 0.2

# Defensive workload proxy
defensive_workload_proxy = (defensive_events / 10.0) + (high_intensity_events / 30.0) + (continuous_possessions / 10.0)

# Fatigue proxy (like Day 8 idea)
fatigue_proxy = (minutes_proxy / 12.0) + (high_intensity_events / 25.0) + (continuous_possessions / 8.0)

# Usage spike proxy
usage_spike_norm = usage_burst  # normalized user input

# Clutch context
is_clutch_context = (period == 4) and (clock_sec <= 300) and (score_diff <= 5)

# -----------------------------
# Calibrate using training distributions (percentiles)
# We map our scenario proxies into the closest existing column distributions
# -----------------------------
usage_col = find_any_col(mart, ["usage_spike_norm"])
def_col = find_any_col(mart, ["defensive_workload_score"])
stab_col = find_any_col(mart, ["lineup_stability_score"])

usage_pct = percentile_rank(mart[usage_col], usage_spike_norm) if usage_col else float("nan")
def_pct = percentile_rank(mart[def_col], defensive_workload_proxy) if def_col else float("nan")
stab_pct = percentile_rank(mart[stab_col], lineup_stability_proxy) if stab_col else float("nan")

# -----------------------------
# Build model input row that matches trained schema
# -----------------------------
# Start from zeros for all expected model features
fatigue_pred = None
fatigue_pred_whatif = None

if fatigue_model is not None:
    # Create empty vector with correct columns
    if hasattr(fatigue_model, "feature_names_in_"):
        cols = list(fatigue_model.feature_names_in_)
        X = pd.DataFrame([{c: 0.0 for c in cols}])
    else:
        # fallback minimal
        X = pd.DataFrame([{}])

    # Try to populate likely columns if they exist
    # These names depend on your feature_mart prefixes, so we do keyword mapping.
    def set_if_exists(keywords, value):
        for c in X.columns:
            if any(k in c.lower() for k in keywords):
                X.loc[0, c] = value

    set_if_exists(["usage_spike_norm"], usage_spike_norm)
    set_if_exists(["defensive_workload"], defensive_workload_proxy)
    set_if_exists(["lineup_stability"], lineup_stability_proxy)
    set_if_exists(["fatigue_score", "fatigue__fatigue_score"], fatigue_proxy)
    set_if_exists(["high_intensity"], float(high_intensity_events))
    set_if_exists(["continuous_possessions"], float(continuous_possessions))
    set_if_exists(["minutes_proxy"], float(minutes_proxy))

    X = X.fillna(0)
    fatigue_pred = float(fatigue_model.predict(X)[0])

    if what_if_rest:
        X2 = X.copy()
        # reduce load-related
        for c in X2.columns:
            cl = c.lower()
            if "usage_spike" in cl or "defensive_workload" in cl or "high_intensity" in cl or "continuous" in cl or "minutes_proxy" in cl:
                X2.loc[0, c] = X2.loc[0, c] * 0.8
        fatigue_pred_whatif = float(fatigue_model.predict(X2)[0])

# Clutch model input
clutch_proba = None
if clutch_model is not None and is_clutch_context:
    # We trained clutch classifier on some basic columns. Provide those if expected.
    if hasattr(clutch_model, "feature_names_in_"):
        cols = list(clutch_model.feature_names_in_)
        Xc = pd.DataFrame([{c: 0.0 for c in cols}])
        # best effort: only clock_sec is known
        if "clock_sec" in Xc.columns:
            Xc.loc[0, "clock_sec"] = float(clock_sec)
        clutch_proba = float(clutch_model.predict_proba(Xc)[:, 1][0])

# -----------------------------
# Alerts + Recommendations (Decision Engine)
# -----------------------------
alerts = []

# Feature alerts
if usage_spike_norm >= 0.6:
    alerts.append("🚨 High usage burst — player is carrying offensive load.")
if def_pct >= 90:
    alerts.append("🚨 Defensive workload proxy is extreme (top 10%).")
if stab_pct <= 25:
    alerts.append("⚠️ Low lineup stability — rotation churn may reduce cohesion.")
if fatigue_proxy >= 1.2:
    alerts.append("⚠️ High fatigue proxy — sustained effort without rest.")

# Model alert
risk_level = "UNKNOWN"
if fatigue_pred is not None:
    # calibrate fatigue_pred roughly via your prediction outputs if available
    past = safe_read_csv(PROJECT_ROOT / "data/outputs/fatigue_predictions.csv")
    if past is not None and "y_pred_drop" in past.columns:
        p = percentile_rank(past["y_pred_drop"], fatigue_pred)
        risk_level = format_level(p)
    alerts.append(f"📉 Fatigue model predicts next-period performance drop: **{fatigue_pred:.2f}** (risk: {risk_level}).")

if fatigue_pred is not None and fatigue_pred_whatif is not None:
    alerts.append(f"🧪 What-if rest simulation reduces predicted drop by **{(fatigue_pred - fatigue_pred_whatif):.2f}**.")

if is_clutch_context:
    alerts.append("⏱️ Clutch context detected (Q4, <=5 min, close score).")
    if clutch_proba is not None:
        alerts.append(f"🎯 Clutch scoring probability (model): **{clutch_proba:.2f}**")

# Recommendations
reco = []

if mode == "Sub/Rest decision":
    if fatigue_proxy >= 1.2 or (fatigue_pred is not None and fatigue_pred > 0):
        reco.append("✅ Consider a short rest/sub: fatigue risk is elevated.")
    if def_pct >= 90:
        reco.append("✅ Reduce defensive burden: switch assignment / send more help / short zone stretch.")
    if stab_pct <= 25:
        reco.append("✅ Stabilize rotation for the next 2–3 possessions (keep core together).")

elif mode == "Offensive usage":
    if usage_spike_norm >= 0.6:
        reco.append("✅ Reduce on-ball load: use off-ball screens, quick hitters, or second-side actions.")
    if recent_turnovers >= 2:
        reco.append("✅ Simplify offense: reduce risky passes, increase spacing and shot quality.")
    if momentum_score < 0:
        reco.append("✅ Stop the bleeding: get to the rim/FT line and slow tempo.")

elif mode == "Clutch shot selection":
    if not is_clutch_context:
        reco.append("ℹ️ Not clutch by definition. Set Q4 + <=300s + diff<=5 for clutch mode.")
    else:
        if clutch_proba is not None and clutch_proba >= 0.6:
            reco.append("✅ Favor this player as a primary finisher in the next action.")
        else:
            reco.append("✅ Prefer a playmaking action: create advantage then kick/secondary attack.")

else:
    reco.append("✅ Use alerts to choose between stabilizing lineups, reducing load, or controlling pace.")
    if momentum_score >= 0.5:
        reco.append("✅ You have momentum — push pace selectively and attack early in the clock.")
    elif momentum_score <= -0.5:
        reco.append("✅ Opponent has momentum — call a set play, value possession, and limit transition.")

# -----------------------------
# UI
# -----------------------------
col1, col2 = st.columns([1.3, 1.0])

with col1:
    st.subheader("📌 Scenario Summary")
    st.write(f"**Period:** {period} | **Clock:** {clock_sec}s | **Score:** {score_home}-{score_away} (diff={score_diff})")
    st.write(f"**Clutch context:** {'YES' if is_clutch_context else 'NO'}")

    st.subheader("🧠 Engineered Signals (scenario-derived)")
    a, b, c = st.columns(3)
    with a:
        st.metric("Usage spike norm", f"{usage_spike_norm:.2f}")
        st.caption(f"Percentile vs training: {usage_pct:.0f} → {format_level(usage_pct)}")
    with b:
        st.metric("Defensive workload proxy", f"{defensive_workload_proxy:.2f}")
        st.caption(f"Percentile vs training: {def_pct:.0f} → {format_level(def_pct)}")
    with c:
        st.metric("Lineup stability proxy", f"{lineup_stability_proxy:.2f}")
        st.caption(f"Percentile vs training: {stab_pct:.0f} → {format_level(stab_pct)}")

    st.subheader("📈 Momentum / Run Context")
    st.write(f"- Recent points margin: **{run_margin:+d}**")
    st.write(f"- Momentum score (proxy): **{momentum_score:.2f}**")
    st.write(f"- Recent turnovers: **{recent_turnovers}**")

    st.subheader("🤖 Model Outputs")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Fatigue drop (pred)", "N/A" if fatigue_pred is None else f"{fatigue_pred:.2f}")
    with m2:
        st.metric("Fatigue proxy", f"{fatigue_proxy:.2f}")
    with m3:
        st.metric("Clutch proba", "N/A" if clutch_proba is None else f"{clutch_proba:.2f}")

    if what_if_rest and fatigue_pred is not None and fatigue_pred_whatif is not None:
        st.subheader("🧪 What-if rest simulation")
        st.write(f"- Baseline drop: **{fatigue_pred:.2f}**")
        st.write(f"- Rest-sim drop: **{fatigue_pred_whatif:.2f}**")
        st.write(f"- Improvement: **{(fatigue_pred - fatigue_pred_whatif):.2f}** (lower drop = better)")

with col2:
    st.subheader("🚨 Alerts")
    if alerts:
        for a in alerts:
            st.write(f"- {a}")
    else:
        st.write("No alerts triggered.")

    st.subheader("✅ Recommendations")
    for r in reco:
        st.write(f"- {r}")

    st.subheader("📝 Coach Notes (copy/paste)")
    notes = []
    notes.append(f"Q{period}, {clock_sec}s, Score {score_home}-{score_away}.")
    notes.append(f"Recent window: margin {run_margin:+d}, TO={recent_turnovers}, shots={recent_shot_attempts}.")
    notes.append(f"Load: usage={usage_spike_norm:.2f}, def_load={defensive_workload_proxy:.2f}, stability={lineup_stability_proxy:.2f}.")
    if fatigue_pred is not None:
        notes.append(f"Fatigue model predicts next-period drop={fatigue_pred:.2f} (risk {risk_level}).")
    if clutch_proba is not None:
        notes.append(f"Clutch scoring probability={clutch_proba:.2f}.")
    if reco:
        notes.append("Recos: " + " ".join([r.replace("✅", "").strip() for r in reco]))

    st.text_area("Notes", "\n".join(notes), height=220)

st.caption("This is a scenario-based assistant. It doesn’t require a game_id; it uses your learned distributions + model schemas to generate decisions.")
