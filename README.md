# ⚡ AI-Powered Smart Controller for a Hybrid Microgrid

> Built during the **Shell-Edunet Skills4Future AICTE Internship** — a 4-week industry-mentored program by Edunet Foundation × AICTE × Shell.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=for-the-badge)](https://console.groq.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

🔗 **[Live Demo →](https://microgrid-controller.streamlit.app/)**

---

## 1. Problem Statement

Solar and wind energy are clean — but they are also unpredictable. The sun doesn't always shine and the wind doesn't always blow at the right time.

Without a smart system to manage this, three big problems happen:

- **Unstable power supply** — too much or too little power at any moment
- **Battery damage** — charging and discharging at the wrong time wears out batteries faster
- **Wasted energy** — extra power gets lost when there is no intelligent storage plan

Old-style controllers follow fixed rules and cannot predict what is coming. This project replaces them with an AI system that looks ahead and makes smarter decisions.

---

## 2. Project Objective

The goal is to build an AI-powered controller that manages a hybrid solar + wind microgrid intelligently.

The system should be able to:

- **Predict** how much solar power, wind power, and electricity demand to expect
- **Decide** in real time whether to charge the battery, discharge it, or exchange power with the main grid
- **Explain** every decision in plain English using an AI language model
- **Detect** if any sensor readings look unusual or suspicious
- **Visualise** everything on a live interactive dashboard

---

## 3. Data Understanding

The dataset covers a full year — **8,758 hours** (Jan to Dec 2023) — and was generated using real physics equations, not random numbers.

Here is what makes the data realistic:

| What it models | How it was done |
|---|---|
| Solar irradiance | Based on actual sun angle (time of day + season) with cloud cover variation |
| Wind speed | Follows the Weibull distribution, an industry-standard model |
| Solar output | Uses 18% panel efficiency with temperature derating (panels lose ~0.4% per °C above 25°C) |
| Wind output | Uses the cubic power law with cut-in (3 m/s), rated (12 m/s), and cut-out (25 m/s) speeds |
| Load demand | Two-peak daily curve — morning (7–9 AM) and evening (6–9 PM) — with seasonal and weekend variation |

The dataset has **21 input features** covering weather, grid state, time of day, and recent history (lag features).

---

## 4. Data Preparation

The data was prepared to be clean and ready for machine learning:

- **Physics-based generation** — data was created using real equations, so there are no fake or random patterns
- **Lag features added** — values from the previous hour (`lag1`) were included so models can learn from recent history
- **Rolling averages added** — 3-hour rolling means were computed for solar, wind, and load
- **Feature scaling** — all features were scaled using `StandardScaler` so no single value dominates the model
- **Train/test split** — 9 months used for training, 3 months held out for testing (no data leakage)

---

## 5. Model Building

Three separate **XGBoost** models were trained — one for each prediction target.

| Model | What it predicts | R² Score | MAE | RMSE |
|---|---|---|---|---|
| `model_solar.joblib` | Solar Output (kW) | **0.9931** | 1.79 kW | 2.64 kW |
| `model_wind.joblib` | Wind Output (kW) | **0.9955** | 3.89 kW | 8.04 kW |
| `model_load.joblib` | Load Demand (kW) | **0.9753** | 11.50 kW | 15.25 kW |

> **What is R²?** It measures prediction accuracy. A score of 1.0 = perfect. A score of 0 = no better than guessing. A negative score = worse than guessing.

The original models in the starter project had **negative R² scores** because the data was randomly generated with no real patterns. By rebuilding the dataset with real physics, all three models now achieve over **97% accuracy**.

**Top features used by each model:**

| Model | Most Important Feature | 2nd Most Important | 3rd |
|---|---|---|---|
| Solar | Solar irradiance (47%) | 3-hour rolling irradiance (39%) | Hour of day (4%) |
| Wind | Wind speed (59%) | Battery charge (13%) | Solar irradiance (7%) |
| Load | 3-hour rolling load (43%) | Hour of day (27%) | Battery charge (8%) |

---

## 6. AI Decision Logic (Groq LLaMA 3.3 70B)

On top of the machine learning predictions, **Groq's LLaMA 3.3 70B** language model is used as an intelligent brain that adds three layers:

**Decision Explanation**
After every simulation, Groq explains in plain English what is happening, why the controller made a specific decision, and any efficiency tips.

**Anomaly Detection**
Groq checks if the sensor readings make sense for the current time and season. For example, if solar output is high at midnight, it will flag that as suspicious.

**Chat Assistant**
Users can type any question about the microgrid and get a clear, contextual answer — like talking to an energy expert.

---

## 7. Dashboard Overview

The app is built with **Streamlit** and runs live in a browser. Here is what you can see and do:

| Feature | What it does |
|---|---|
| **Sidebar controls** | Choose the month, hour, temperature, wind speed, and solar irradiance |
| **KPI cards** | Shows predicted solar output, wind output, load demand, and net energy flow |
| **Energy Balance Chart** | Stacked bar chart comparing total generation vs demand |
| **24-Hour Profile** | Average daily curve for the selected month with a "Now" marker |
| **Grid Status Banner** | Green (stable, surplus) or Red (stressed, deficit) |
| **Net Flow Gauge** | Visual dial showing how much power is surplus or missing |
| **Groq AI Explanation** | Plain-English analysis of the current grid state |
| **Anomaly Alert** | Flags anything unusual about the sensor readings |
| **Chat Assistant** | Ask any question about the microgrid in natural language |
| **Quick Question Buttons** | One-click preset questions for common queries |

---

## 8. Key Insights

After analysing the data and running the system, here are the main findings:

- **Solar irradiance is the dominant factor** for solar prediction — it alone accounts for 47% of the model's decisions, with the 3-hour rolling average adding another 39%
- **Wind speed follows a cubic relationship** with power output — small changes in wind speed have a large impact on generation
- **Evening hours (6–9 PM) create the highest grid stress** — load demand peaks while solar output drops to zero
- **Battery management is critical in winter** — shorter days and lower irradiance mean solar generates very little, and the battery must bridge the gap
- **Weekend load patterns differ noticeably** from weekday patterns, which the model learned from the `day_of_week` feature
- **The original models were completely broken** — rebuilding the dataset with physics-based generation improved R² from negative values to 0.97–0.99 across all three targets

---

## 9. Recommendations

Based on the project insights, here are suggestions for improving the system further:

- **Connect to a real weather API** (like OpenWeatherMap or Open-Meteo) so predictions use actual forecasts instead of sliders
- **Add battery health modelling** — track State of Health (SoH) over time so the system avoids decisions that degrade the battery faster
- **Enable multi-day forecasting** — extend predictions to 24h, 48h, and 72h ahead for better planning
- **Add cost optimisation** — use time-of-use electricity tariffs to decide the cheapest time to import from or export to the main grid
- **Compare with LSTM or Transformer models** — XGBoost performs well here, but deep learning models may capture longer time patterns better
- **Set up alerts** — send an email or SMS notification when the grid enters a stress state

---

## 10. Skills Demonstrated

This project covers a wide range of data science, machine learning, and software engineering skills:

- **Data Engineering** — physics-based synthetic data generation, feature engineering, lag and rolling window features
- **Machine Learning** — XGBoost regression, train/test splitting, feature importance analysis, model evaluation (R², MAE, RMSE)
- **Data Preprocessing** — StandardScaler normalisation, handling NaN values, consistent feature alignment
- **AI Integration** — using Groq's LLaMA 3.3 70B for decision explanation, anomaly detection, and natural language Q&A
- **Dashboard Development** — building a live interactive app with Streamlit and Plotly (gauges, stacked bars, line charts)
- **Software Engineering** — modular Python code, error handling, model serialisation with joblib, secrets management
- **Deployment** — hosting the app on Streamlit Cloud with environment variable configuration

---

## 11. Files in This Repository

```
microgrid-controller/
│
├── app.py                    → Main Streamlit dashboard (the entire application)
├── requirements.txt          → Python packages needed to run the project
│
├── processed_dataset.csv     → Physics-based training dataset (8,758 rows, full year)
├── model_solar.joblib        → Trained solar power prediction model (R² = 0.9931)
├── model_wind.joblib         → Trained wind power prediction model  (R² = 0.9955)
├── model_load.joblib         → Trained load demand prediction model (R² = 0.9753)
├── scaler.joblib             → StandardScaler fitted on the training data
├── model_metadata.json       → Feature column names and training information
│
├── generate_dataset.py       → Script to regenerate the dataset from scratch (run locally)
└── train_models.py           → Script to retrain all three models (run locally)
```

---

## 12. How to Run This Project

### Option A — Run Locally

**Step 1 — Clone the repository**
```bash
git clone https://github.com/DHAMMADEEPRAMTEKE30/microgrid-controller.git
cd microgrid-controller
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Add your Groq API key**

Create a file at `.streamlit/secrets.toml` and paste:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
Get a free key at [console.groq.com](https://console.groq.com) — takes under 2 minutes.

**Step 4 — Start the app**
```bash
streamlit run app.py
```

---

### Option B — Deploy on Streamlit Cloud (Free)

1. Fork this repository on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App → Select your repo → Set main file to `app.py`
3. In App Settings → Secrets → paste your `GROQ_API_KEY`
4. Set Python version to **3.11** in Advanced Settings
5. Click **Deploy** — your app will be live in about 2 minutes

---

### Optional — Regenerate Data or Retrain Models

If you want to rebuild everything from scratch:
```bash
python generate_dataset.py   # Rebuilds processed_dataset.csv
python train_models.py       # Retrains and saves all .joblib model files
```

---

## Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.11 |
| Dashboard | Streamlit |
| Machine Learning | XGBoost + Scikit-learn |
| AI Brain | Groq API — LLaMA 3.3 70B |
| Charts | Plotly |
| Data Handling | Pandas, NumPy |
| Deployment | Streamlit Cloud |

---

## Acknowledgements

- **Edunet Foundation** — for organising the Shell-Edunet Skills4Future AICTE Internship
- **Shell** — for industry sponsorship and mentorship
- **AICTE** — for program accreditation and certification
- **Groq** — for fast LLM inference powering the AI brain
- **XGBoost team** — for the gradient boosting library

---

## 👤 Author

**Dhammadeep Anil Ramteke**

- 💼 LinkedIn: https://www.linkedin.com/in/dhammadeep-ramteke/
- 🐙 GitHub: https://github.com/DHAMMADEEPRAMTEKE30
- 📧 Email: ramtekedhamma30@gmail.com / dhammadeepramteke2702@gmail.com

---

*Built with ❤️ during the Shell-Edunet Skills4Future AICTE Internship · 2025*
