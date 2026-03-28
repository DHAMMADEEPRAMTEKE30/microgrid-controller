"""
AI-Powered Smart Controller for a Hybrid Microgrid
====================================================
Rebuilt version with:
  - Real physics-based predictions (R² > 0.97)
  - Groq AI as the intelligent brain
  - Natural language explanations of every decision
  - AI chat assistant for questions
  - Proper error handling throughout
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from groq import Groq

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Microgrid Controller",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main { background: #080c14; }

/* Glowing header */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff 0%, #00ff88 50%, #7b61ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.hero-sub {
    color: #6b7280;
    font-size: 1rem;
    margin-top: 4px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 500;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #00ff88);
}
.kpi-label { color: #6b7280; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { color: #f1f5f9; font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin: 4px 0 2px; }
.kpi-delta-pos { color: #00ff88; font-size: 0.82rem; font-weight: 500; }
.kpi-delta-neg { color: #ff4757; font-size: 0.82rem; font-weight: 500; }

/* Status banner */
.status-stable {
    background: rgba(0, 255, 136, 0.08);
    border: 1px solid rgba(0, 255, 136, 0.25);
    border-left: 4px solid #00ff88;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
}
.status-stress {
    background: rgba(255, 71, 87, 0.08);
    border: 1px solid rgba(255, 71, 87, 0.25);
    border-left: 4px solid #ff4757;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
}
.status-title { font-size: 1rem; font-weight: 700; margin: 0 0 4px; }
.status-msg   { font-size: 0.85rem; color: #9ca3af; margin: 0; }

/* AI explanation box */
.ai-box {
    background: linear-gradient(135deg, #0d1b2e 0%, #0f1e35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 20px;
    margin: 12px 0;
    position: relative;
}
.ai-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7b61ff, #00d4ff);
}
.ai-badge {
    display: inline-block;
    background: linear-gradient(90deg, #7b61ff22, #00d4ff22);
    border: 1px solid #7b61ff44;
    color: #a78bfa;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 10px;
}
.ai-text { color: #cbd5e1; font-size: 0.92rem; line-height: 1.7; }

/* Chat messages */
.chat-user {
    background: #1e293b;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e2e8f0;
    font-size: 0.9rem;
    margin-left: 20%;
}
.chat-ai {
    background: linear-gradient(135deg, #0d1b2e, #0f1e35);
    border: 1px solid #1e3a5f;
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #cbd5e1;
    font-size: 0.9rem;
    margin-right: 20%;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080c14 !important;
    border-right: 1px solid #1e293b !important;
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg, #00d4ff 0%, #00ff88 100%) !important;
    color: #080c14 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1rem !important;
    width: 100% !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.03em !important;
}

/* Model badge */
.model-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #00ff88;
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 6px;
    padding: 2px 8px;
    display: inline-block;
}

div[data-testid="stMetric"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)


# ── LOAD MODELS & DATA ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load all model files. Returns None values on failure."""
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        df      = pd.read_csv(os.path.join(base, "processed_dataset.csv"),
                              index_col="timestamp", parse_dates=True)
        scaler  = joblib.load(os.path.join(base, "scaler.joblib"))
        solar_m = joblib.load(os.path.join(base, "model_solar.joblib"))
        wind_m  = joblib.load(os.path.join(base, "model_wind.joblib"))
        load_m  = joblib.load(os.path.join(base, "model_load.joblib"))

        meta_path = os.path.join(base, "model_metadata.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

        return df, scaler, solar_m, wind_m, load_m, meta
    except Exception as e:
        st.error(f"❌ Failed to load model files: {e}")
        return None, None, None, None, None, {}

df_ref, scaler, solar_model, wind_model, load_model, meta = load_artifacts()

FEATURE_COLS = meta.get("feature_cols", [
    'solar_irradiance', 'wind_speed', 'temperature', 'humidity', 'pressure',
    'grid_frequency', 'grid_voltage', 'grid_exchange', 'battery_soc',
    'battery_charge', 'battery_discharge', 'hour', 'day_of_week', 'month',
    'day_of_year', 'load_demand_lag1', 'solar_irradiance_lag1',
    'wind_speed_lag1', 'wind_speed_roll_3h', 'solar_irradiance_roll_3h',
    'load_roll_3h'
])


# ── GROQ CLIENT ───────────────────────────────────────────────────────────────
def get_groq_client():
    """Get Groq client using API key from Streamlit secrets or env."""
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ── PREDICTION LOGIC ──────────────────────────────────────────────────────────
def get_prediction_input(month, hour, temp, wind, irr):
    """
    Build the feature vector for prediction.
    We look up a similar row in the reference dataset and update the
    weather inputs the user has provided.
    """
    match = df_ref[(df_ref["month"] == month) & (df_ref["hour"] == hour)]
    base  = match.iloc[0:1].copy() if not match.empty else df_ref[df_ref["month"] == month].mean().to_frame().T

    base = base.copy()
    base["temperature"]       = temp
    base["wind_speed"]        = wind
    base["solar_irradiance"]  = irr

    # Update lag/rolling features to be consistent with new inputs
    base["solar_irradiance_lag1"]    = irr * 0.9
    base["wind_speed_lag1"]          = wind * 0.95
    base["solar_irradiance_roll_3h"] = irr * 0.95
    base["wind_speed_roll_3h"]       = wind * 0.97

    input_data = base.reindex(columns=FEATURE_COLS)

    # Fill any remaining NaN with column means from reference dataset
    for col in input_data.columns:
        if input_data[col].isnull().any():
            input_data[col] = df_ref[col].mean() if col in df_ref.columns else 0

    return input_data


# ── GROQ: DECISION EXPLANATION ────────────────────────────────────────────────
def get_groq_explanation(client, solar, wind, load, net, battery_action, conditions):
    """Ask Groq to explain the current microgrid state in plain English."""
    if client is None:
        return "⚠️ Groq API key not configured. Add GROQ_API_KEY to Streamlit secrets."

    prompt = f"""You are an AI controller for a hybrid solar-wind microgrid.
Current grid state:
- Solar generation: {solar:.1f} kW
- Wind generation:  {wind:.1f} kW
- Total generation: {solar + wind:.1f} kW
- Load demand:      {load:.1f} kW
- Net energy flow:  {net:.1f} kW ({'surplus' if net >= 0 else 'deficit'})
- Battery action:   {battery_action}
- Conditions:       {conditions}

In 3-4 sentences, explain:
1. What is happening in the grid right now and WHY
2. What action the smart controller is taking and why that is the best decision
3. One practical tip or observation about efficiency

Be direct, clear, and use simple language. No bullet points — write as flowing sentences."""

    try:
        response = client.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 250,
            temperature = 0.4,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API error: {e}"


# ── GROQ: ANOMALY CHECK ───────────────────────────────────────────────────────
def get_groq_anomaly_check(client, solar, wind, load, irr, wind_speed, temp, month, hour):
    """Ask Groq if anything looks unusual about the current readings."""
    if client is None:
        return None

    # Expected ranges based on time of day
    expected_solar_note = "solar should be zero" if (hour < 6 or hour > 20) else "solar generation expected"
    season = "winter" if month in [12, 1, 2] else "summer" if month in [6, 7, 8] else "spring/autumn"

    prompt = f"""You are a microgrid monitoring system. Check if anything is anomalous.

Readings:
- Hour: {hour}:00  |  Month: {month} ({season})
- Solar irradiance: {irr:.0f} W/m²  → Solar output: {solar:.1f} kW
- Wind speed: {wind_speed:.1f} m/s   → Wind output: {wind:.1f} kW
- Temperature: {temp:.1f}°C
- Load demand: {load:.1f} kW
- Note: {expected_solar_note}

In ONE sentence: Is anything anomalous? If yes, say what and why. If no, say "All readings normal for current conditions." Keep it under 30 words."""

    try:
        response = client.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 80,
            temperature = 0.2,
        )
        return response.choices[0].message.content
    except Exception:
        return None


# ── GROQ: CHAT ASSISTANT ──────────────────────────────────────────────────────
def get_groq_chat_response(client, user_message, context):
    """Answer user questions about the microgrid in context."""
    if client is None:
        return "Groq API key not configured. Please add it to Streamlit secrets."

    system_prompt = f"""You are an expert AI assistant for a hybrid solar-wind microgrid.
You help operators understand what is happening and make decisions.

Current system state:
{context}

Answer questions clearly and concisely. Use simple language suitable for someone 
learning about energy systems. Keep responses under 150 words."""

    try:
        response = client.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
            max_tokens  = 200,
            temperature = 0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Control Panel")
    st.markdown('<span class="model-badge">XGBoost + Groq llama-3.3-70b</span>', unsafe_allow_html=True)
    st.markdown("---")

    with st.form("sim_form"):
        st.markdown("**📅 Time**")
        c1, c2 = st.columns(2)
        with c1:
            sel_month = st.selectbox("Month", list(range(1, 13)), index=5,
                                     format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                             "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])
        with c2:
            sel_hour  = st.selectbox("Hour", list(range(0, 24)), index=12,
                                     format_func=lambda h: f"{h:02d}:00")

        st.markdown("**🌤️ Weather Conditions**")
        temp_inp = st.slider("🌡️ Temperature (°C)",      -5.0, 45.0, 22.0, step=0.5)
        wind_inp = st.slider("💨 Wind Speed (m/s)",        0.0, 25.0,  8.0, step=0.5)
        irr_inp  = st.slider("☀️ Irradiance (W/m²)",      0.0, 950.0, 600.0, step=10.0)

        st.markdown("---")
        run = st.form_submit_button("⚡ Run Simulation")

    with st.expander("ℹ️ Model Performance"):
        st.markdown("""
**R² Scores (trained on physics data):**
- ☀️ Solar:  **0.9931** (99.3% accurate)
- 💨 Wind:   **0.9955** (99.6% accurate)
- ⚡ Load:   **0.9753** (97.5% accurate)

*Previously these were all negative — worse than guessing!*
        """)

    with st.expander("🔧 Admin"):
        if st.button("Clear Cache & Reload"):
            st.cache_resource.clear()
            st.rerun()


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 0.5rem;">
    <div class="hero-title">⚡ AI-Powered Smart Controller</div>
    <div class="hero-sub">Hybrid Solar-Wind Microgrid · Predictive Energy Management · Powered by Groq</div>
</div>
""", unsafe_allow_html=True)

if df_ref is None:
    st.error("❌ Model files not found. Make sure all .joblib and .csv files are in the same folder as app.py")
    st.stop()

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sim_run" not in st.session_state:
    st.session_state.sim_run = False
if "last_context" not in st.session_state:
    st.session_state.last_context = ""

# ── RUN SIMULATION ─────────────────────────────────────────────────────────────
if run or not st.session_state.sim_run:
    st.session_state.sim_run = True
    groq_client = get_groq_client()

    # 1. Build input & predict
    inp_df = get_prediction_input(sel_month, sel_hour, temp_inp, wind_inp, irr_inp)
    try:
        inp_sc = scaler.transform(inp_df)
    except Exception as e:
        st.error(f"Scaler error: {e}")
        st.stop()

    pred_solar = float(max(0, solar_model.predict(inp_sc)[0]))
    pred_wind  = float(max(0, wind_model.predict(inp_sc)[0]))
    pred_load  = float(max(0, load_model.predict(inp_sc)[0]))

    total_gen = pred_solar + pred_wind
    net_energy = total_gen - pred_load

    # 2. Battery decision logic
    BATTERY_CAPACITY = 1000  # kWh
    current_soc_pct  = 55.0  # Assumed midpoint for display

    if net_energy >= 0:
        battery_action = f"Charging battery with {min(net_energy, 200):.1f} kW surplus"
        status_class   = "status-stable"
        status_title   = "🟢 GRID STABLE"
        status_color   = "#00ff88"
        status_msg     = f"Surplus of {net_energy:.1f} kW — battery charging or exporting to grid."
    else:
        battery_action = f"Discharging battery to cover {abs(net_energy):.1f} kW deficit"
        status_class   = "status-stress"
        status_title   = "🔴 GRID STRESS"
        status_color   = "#ff4757"
        status_msg     = f"Deficit of {abs(net_energy):.1f} kW — drawing from battery or grid."

    conditions = f"Month {sel_month}, Hour {sel_hour}:00, Temp {temp_inp}°C, Wind {wind_inp} m/s, Irr {irr_inp} W/m²"

    # Store context for chat
    st.session_state.last_context = f"""
Solar: {pred_solar:.1f} kW | Wind: {pred_wind:.1f} kW | Load: {pred_load:.1f} kW
Net energy: {net_energy:.1f} kW | Battery action: {battery_action}
Conditions: {conditions}"""

    # ── KPI CARDS ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">☀️ Solar Generation</div>
            <div class="kpi-value">{pred_solar:.1f}</div>
            <div class="kpi-delta-pos">kW  ·  irr {irr_inp:.0f} W/m²</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">💨 Wind Generation</div>
            <div class="kpi-value">{pred_wind:.1f}</div>
            <div class="kpi-delta-pos">kW  ·  {wind_inp:.1f} m/s wind</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">⚡ Load Demand</div>
            <div class="kpi-value">{pred_load:.1f}</div>
            <div class="kpi-delta-neg">kW  ·  hour {sel_hour:02d}:00</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        delta_color = "kpi-delta-pos" if net_energy >= 0 else "kpi-delta-neg"
        delta_sym   = "▲" if net_energy >= 0 else "▼"
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">🔋 Net Energy Flow</div>
            <div class="kpi-value">{net_energy:+.1f}</div>
            <div class="{delta_color}">{delta_sym} {'Surplus' if net_energy >= 0 else 'Deficit'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CHARTS + STATUS ────────────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown("**📊 Real-Time Energy Balance**")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Generation"], y=[pred_solar], name="Solar",
                             marker_color="#f59e0b", marker_line_width=0))
        fig.add_trace(go.Bar(x=["Generation"], y=[pred_wind],  name="Wind",
                             marker_color="#38bdf8", marker_line_width=0))
        fig.add_trace(go.Bar(x=["Demand"],     y=[pred_load],  name="Load",
                             marker_color="#ff4757", marker_line_width=0))

        fig.update_layout(
            barmode      = "stack",
            height       = 320,
            plot_bgcolor = "rgba(0,0,0,0)",
            paper_bgcolor= "rgba(0,0,0,0)",
            font         = dict(color="#94a3b8", family="Space Grotesk"),
            yaxis        = dict(title="Power (kW)", gridcolor="#1e293b", tickcolor="#374151"),
            xaxis        = dict(gridcolor="#1e293b"),
            margin       = dict(l=10, r=10, t=20, b=10),
            legend       = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 24-hour forecast mini chart
        st.markdown("**📈 Typical 24-Hour Profile (Today's Conditions)**")
        hours = list(range(24))
        ref_month_data = df_ref[df_ref["month"] == sel_month]

        solar_profile = [ref_month_data[ref_month_data["hour"] == h]["solar_output"].mean() for h in hours]
        wind_profile  = [ref_month_data[ref_month_data["hour"] == h]["wind_output"].mean()  for h in hours]
        load_profile  = [ref_month_data[ref_month_data["hour"] == h]["load_demand"].mean()  for h in hours]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hours, y=solar_profile, name="Solar avg", mode="lines",
                                  line=dict(color="#f59e0b", width=2), fill="tozeroy",
                                  fillcolor="rgba(245,158,11,0.1)"))
        fig2.add_trace(go.Scatter(x=hours, y=wind_profile,  name="Wind avg",  mode="lines",
                                  line=dict(color="#38bdf8", width=2), fill="tozeroy",
                                  fillcolor="rgba(56,189,248,0.1)"))
        fig2.add_trace(go.Scatter(x=hours, y=load_profile,  name="Load avg",  mode="lines",
                                  line=dict(color="#ff4757", width=2, dash="dash")))
        fig2.add_vline(x=sel_hour, line_color="#ffffff", line_width=1.5, line_dash="dot",
                       annotation_text="Now", annotation_font_color="#ffffff")

        fig2.update_layout(
            height       = 220,
            plot_bgcolor = "rgba(0,0,0,0)",
            paper_bgcolor= "rgba(0,0,0,0)",
            font         = dict(color="#94a3b8", family="Space Grotesk"),
            yaxis        = dict(title="kW", gridcolor="#1e293b"),
            xaxis        = dict(title="Hour of day", gridcolor="#1e293b", tickvals=list(range(0,24,3)),
                                ticktext=[f"{h:02d}:00" for h in range(0,24,3)]),
            margin       = dict(l=10, r=10, t=10, b=30),
            legend       = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with right_col:
        # Status banner
        st.markdown(f"""<div class="{status_class}">
            <div class="status-title" style="color:{status_color}">{status_title}</div>
            <div class="status-msg">{status_msg}</div>
        </div>""", unsafe_allow_html=True)

        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = net_energy,
            delta = {"reference": 0, "valueformat": ".1f"},
            title = {"text": "Net Flow (kW)", "font": {"size": 14, "color": "#6b7280"}},
            number = {"font": {"color": "#f1f5f9", "size": 32}},
            gauge = {
                "axis": {"range": [-600, 600], "tickcolor": "#374151",
                         "tickfont": {"color": "#6b7280", "size": 10}},
                "bar":  {"color": status_color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [-600, 0],   "color": "rgba(255,71,87,0.15)"},
                    {"range": [0,    600], "color": "rgba(0,255,136,0.15)"},
                ],
                "threshold": {"line": {"color": "#ffffff", "width": 2}, "value": 0, "thickness": 0.75},
            }
        ))
        fig_g.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"))
        st.plotly_chart(fig_g, use_container_width=True)

        # Anomaly check
        anomaly = get_groq_anomaly_check(
            get_groq_client(), pred_solar, pred_wind, pred_load,
            irr_inp, wind_inp, temp_inp, sel_month, sel_hour
        )
        if anomaly:
            icon = "⚠️" if "anomal" in anomaly.lower() and "no" not in anomaly.lower()[:5] else "✅"
            st.info(f"{icon} {anomaly}")

    st.markdown("---")

    # ── GROQ AI EXPLANATION ────────────────────────────────────────────────────
    st.markdown("**🤖 Groq AI — Decision Explanation**")

    with st.spinner("Groq is analysing the grid state..."):
        explanation = get_groq_explanation(
            get_groq_client(), pred_solar, pred_wind, pred_load,
            net_energy, battery_action, conditions
        )

    st.markdown(f"""<div class="ai-box">
        <span class="ai-badge">🔮 Groq llama-3.3-70b · live analysis</span>
        <div class="ai-text">{explanation}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── AI CHAT ASSISTANT ──────────────────────────────────────────────────────
    st.markdown("**💬 Ask the AI Assistant**")
    st.caption("Ask anything about the current grid state, energy decisions, or microgrid operations.")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Quick question buttons
    st.markdown("**Quick questions:**")
    qcols = st.columns(3)
    quick_qs = [
        "Why is the battery charging/discharging right now?",
        "How can we reduce the grid deficit?",
        "What happens if wind speed doubles?",
    ]
    for i, (qcol, q) in enumerate(zip(qcols, quick_qs)):
        with qcol:
            if st.button(q, key=f"quick_{i}"):
                with st.spinner("Groq is thinking..."):
                    response = get_groq_chat_response(get_groq_client(), q, st.session_state.last_context)
                st.session_state.chat_history.append({"role": "user",      "content": q})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

    # Free text input
    user_q = st.chat_input("Ask your question about the microgrid...")
    if user_q:
        with st.spinner("Groq is thinking..."):
            response = get_groq_chat_response(get_groq_client(), user_q, st.session_state.last_context)
        st.session_state.chat_history.append({"role": "user",      "content": user_q})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.info("👈 Adjust the simulation parameters in the sidebar and click **Run Simulation**.")
