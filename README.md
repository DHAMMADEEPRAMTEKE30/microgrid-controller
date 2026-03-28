# ⚡ AI-Powered Smart Controller for a Hybrid Microgrid

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189AB4?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade, AI-powered predictive energy management system for a hybrid solar-wind microgrid — built during the Shell-Edunet Skills4Future AICTE Internship.**

[🚀 Live Demo](https://microgrid-controller.streamlit.app/)

</div>

---

## 🎓 Internship Background

This project was developed as part of the **Shell-Edunet Skills4Future AICTE Internship**, a 4-week industry-mentored program run by the **Edunet Foundation** in collaboration with **AICTE** (All India Council for Technical Education) and **Shell**.

| Detail | Info |
|---|---|
| **Program** | Shell-Edunet Skills4Future AICTE Internship |
| **Presented By** | Edunet Foundation × AICTE × Shell |
| **Focus Area** | Green Skills using AI Technologies |
| **Duration** | 4 Weeks — 27th October 2025 to 27th November 2025 |
| **Project Theme** | Energy Prediction using Machine Learning |
| **Certification** | AICTE · Shell · Edunet |

The program provides participants with hands-on, project-based learning under the guidance of industry mentors, with the goal of developing real-world AI solutions for sustainability challenges.

> *"Using historical data and machine learning algorithms to forecast future outcomes — such as solar energy forecasting."*

---

## 📌 Project Overview

Renewable energy systems like solar and wind are inherently **unpredictable** — the sun doesn't always shine and the wind doesn't always blow. Without intelligent management, this leads to power instability, battery degradation, and energy wastage.

This project builds an **AI-powered Smart Controller** that:

- 🔮 **Forecasts** solar generation, wind generation, and electricity demand using XGBoost models trained on physics-based data
- 🧠 **Decides** in real time whether to charge/discharge the battery or exchange with the main grid
- 💬 **Explains** every decision in plain English using **Groq's LLaMA 3.3 70B** large language model
- 📊 **Visualises** the energy balance, 24-hour profiles, and grid status on a live interactive dashboard

---

## 🚨 Problem Statement

Hybrid renewable energy systems face three core challenges:

1. **Power Instability** — unpredictable generation from solar and wind
2. **Battery Stress** — poor charge/discharge decisions degrade the battery faster
3. **Energy Wastage** — surplus energy is lost when there is no intelligent storage management

Traditional rule-based controllers cannot anticipate these fluctuations. This project replaces them with a **predictive, AI-driven approach**.

---

## 🎯 Project Objectives

| Objective | Description |
|---|---|
| **Forecast** | Predict solar power, wind power, and load demand using ML models |
| **Control** | Manage power flow between solar, wind, battery, and the grid |
| **Optimise** | Make real-time decisions to reduce grid dependency and battery wear |
| **Explain** | Use Groq AI to explain every decision in natural language |
| **Visualise** | Display live energy balance, status, and trends on a dashboard |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │  Solar   │  │   Wind   │  │   Load   │  │  Net   │  │
│  │  KPI     │  │   KPI    │  │   KPI    │  │ Flow   │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
│         ↑              ↑              ↑                  │
│  ┌──────────────────────────────────────────────────┐   │
│  │           XGBoost Prediction Engine              │   │
│  │   model_solar.joblib  ·  model_wind.joblib       │   │
│  │              model_load.joblib                   │   │
│  └──────────────────────────────────────────────────┘   │
│         ↑                            ↓                   │
│  ┌────────────────┐    ┌─────────────────────────────┐  │
│  │ Weather Inputs │    │     Groq AI Brain           │  │
│  │ · Temperature  │    │  · Decision Explanation     │  │
│  │ · Wind Speed   │    │  · Anomaly Detection        │  │
│  │ · Irradiance   │    │  · Chat Assistant (Q&A)     │  │
│  └────────────────┘    └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🤖 AI Models

### Prediction Models (XGBoost)

Three independent XGBoost regressors were trained on **8,758 hours** of physics-based synthetic data covering a full year (Jan–Dec 2023), with a 9-month/3-month train-test split.

| Model | Target | R² Score | MAE | RMSE |
|---|---|---|---|---|
| `model_solar.joblib` | Solar Output (kW) | **0.9931** | 1.79 kW | 2.64 kW |
| `model_wind.joblib` | Wind Output (kW) | **0.9955** | 3.89 kW | 8.04 kW |
| `model_load.joblib` | Load Demand (kW) | **0.9753** | 11.50 kW | 15.25 kW |

> An R² score of **1.0 = perfect prediction**, **0 = no better than guessing the average**, **< 0 = worse than useless**.

### What Makes This Data Realistic

The dataset was generated using real physics equations — not random numbers:

- **Solar irradiance** follows the actual sun angle (time of day + season), with cloud cover variability using a beta distribution
- **Wind speed** follows the industry-standard Weibull distribution with seasonal variation
- **Solar output** uses a real 18% panel efficiency model with temperature derating (panels lose ~0.4% efficiency per °C above 25°C)
- **Wind output** uses the real **cubic power law** with cut-in (3 m/s), rated (12 m/s), and cut-out (25 m/s) speeds
- **Load demand** follows a real two-hump daily curve (morning peak 7–9 AM, evening peak 6–9 PM) with seasonal and weekend variation

### Input Features (21 total)

```
Weather:    solar_irradiance, wind_speed, temperature, humidity, pressure
Grid:       grid_frequency, grid_voltage, grid_exchange, battery_soc,
            battery_charge, battery_discharge
Time:       hour, day_of_week, month, day_of_year
Lag:        load_demand_lag1, solar_irradiance_lag1, wind_speed_lag1
Rolling:    wind_speed_roll_3h, solar_irradiance_roll_3h, load_roll_3h
```

### Top Feature Importance

| Model | #1 Feature | #2 Feature | #3 Feature |
|---|---|---|---|
| Solar | `solar_irradiance` (47%) | `solar_irradiance_roll_3h` (39%) | `hour` (4%) |
| Wind | `wind_speed` (59%) | `battery_charge` (13%) | `solar_irradiance` (7%) |
| Load | `load_roll_3h` (43%) | `hour` (27%) | `battery_charge` (8%) |

### Groq AI Brain (LLaMA 3.3 70B)

The Groq integration adds three intelligent layers on top of the ML predictions:

1. **Decision Explanation** — After every simulation, Groq explains in plain English why the grid is stable or stressed, what the controller is doing, and any efficiency tips
2. **Anomaly Detection** — Groq checks if sensor readings make sense for the time of day and season (e.g., high solar output at midnight would be flagged)
3. **Chat Assistant** — Users can ask any question about the microgrid in natural language and get a contextual answer

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.11 |
| **Dashboard** | [Streamlit](https://streamlit.io/) |
| **ML Models** | [XGBoost](https://xgboost.readthedocs.io/) + [Scikit-learn](https://scikit-learn.org/) |
| **AI Brain** | [Groq API](https://console.groq.com/) — LLaMA 3.3 70B Versatile |
| **Visualisation** | [Plotly](https://plotly.com/python/) |
| **Data** | Pandas, NumPy |
| **Deployment** | [Streamlit Cloud](https://share.streamlit.io/) |

---

## 📁 File Structure

```
microgrid-controller/
│
├── app.py                    # Main Streamlit dashboard
├── requirements.txt          # Python dependencies
│
├── processed_dataset.csv     # Physics-based training dataset (8,758 rows)
├── model_solar.joblib        # Trained solar prediction model (R² = 0.9931)
├── model_wind.joblib         # Trained wind prediction model  (R² = 0.9955)
├── model_load.joblib         # Trained load prediction model  (R² = 0.9753)
├── scaler.joblib             # StandardScaler (fit on training data)
├── model_metadata.json       # Feature columns + training info
│
├── generate_dataset.py       # [Local only] Physics-based data generator
├── train_models.py           # [Local only] Model training script
│
└── .streamlit/
    └── secrets.toml          # [Local only — NEVER commit] API keys
```

---

## 🚀 Getting Started

### Option A — Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/microgrid-controller.git
cd microgrid-controller
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Groq API key**

Create a file at `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
Get your free API key at [console.groq.com](https://console.groq.com) — it takes under 2 minutes.

**4. Run the app**
```bash
streamlit run app.py
```

### Option B — Deploy on Streamlit Cloud

**1.** Fork this repository on GitHub

**2.** Go to [share.streamlit.io](https://share.streamlit.io) → New App → Select your repo → Set main file to `app.py`

**3.** In the app settings → **Secrets** → paste:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

**4.** Set Python version to **3.11** in Advanced Settings

**5.** Click **Deploy** — your app will be live in ~2 minutes ✅

### Regenerate the Dataset or Retrain Models (Optional)

If you want to regenerate the data from scratch or retrain the models:
```bash
python generate_dataset.py   # Regenerates processed_dataset.csv
python train_models.py       # Retrains and saves all .joblib files
```

---

## 📊 Dashboard Features

| Feature | Description |
|---|---|
| **KPI Cards** | Live solar, wind, load, and net energy values |
| **Energy Balance Chart** | Stacked bar — generation vs demand |
| **24-Hour Profile** | Average daily curve for the selected month with "now" marker |
| **Grid Status Banner** | Green (stable) or Red (stress) with kW values |
| **Net Flow Gauge** | Visual indicator of surplus/deficit |
| **Groq AI Explanation** | Plain-English analysis of every simulation result |
| **Anomaly Alert** | Groq flags anything unusual about the sensor readings |
| **Chat Assistant** | Ask any question about the microgrid in natural language |
| **Quick Questions** | One-click preset questions for common queries |
| **Sidebar Controls** | Month, hour, temperature, wind speed, irradiance sliders |

---

## 📈 Results & Performance

### Model Accuracy Improvement

| Metric | Original Project | This Project |
|---|---|---|
| Solar R² Score | -0.013 ❌ | **0.9931** ✅ |
| Wind R² Score | -0.006 ❌ | **0.9955** ✅ |
| Load R² Score | 0.436 ⚠️ | **0.9753** ✅ |
| AI Explanation | Hardcoded fake | **Live Groq LLM** ✅ |
| Chat Assistant | None | **Full NL Q&A** ✅ |

The original models had **negative R² scores** because the training data was randomly generated with no real physical patterns. By rebuilding the dataset using actual physics equations (solar angle, wind power curve, load demand profiles), the models now achieve **>97% accuracy** across all three targets.

---

## 🔮 Future Improvements

- [ ] Integrate real-time weather API (OpenWeatherMap or Open-Meteo)
- [ ] Add battery State of Health (SoH) degradation modelling
- [ ] Multi-day ahead forecasting (24h, 48h, 72h)
- [ ] Cost optimisation — minimise electricity bill using time-of-use tariffs
- [ ] LSTM / Transformer model comparison against XGBoost
- [ ] Alert system via email/SMS when grid stress is detected

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **Edunet Foundation** — for organising the Shell-Edunet Skills4Future AICTE Internship
- **Shell** — for industry sponsorship and mentorship
- **AICTE** — for program accreditation and certification
- **Groq** — for blazing-fast LLM inference that powers the AI brain
- **XGBoost team** — for the world-class gradient boosting library

---

<div align="center">

**Built with ❤️ during the Shell-Edunet Skills4Future AICTE Internship · 2025**

[⬆ Back to top](#-ai-powered-smart-controller-for-a-hybrid-microgrid)

</div>
