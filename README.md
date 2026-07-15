# 🌡️ Delhi Urban Heat Island — AI/ML Mitigation Dashboard

> **Physics-informed Explainable AI System for Urban Heat Stress Mapping, Hotspot Detection and Cooling Intervention Planning for Delhi NCR**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-success)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Live Dashboard

👉 https://delhi-uhi-dashboard.streamlit.app

---

## 📸 Dashboard Preview

### Project Overview

![Overview](images/overview.png)

### Heat Stress Mapping

![Heat Stress Map](images/heat_stress_map.png)

### Hotspot Detection

![Hotspot Map](images/hotspot_map.png)

### Cooling Effect Map

![Cooling Effect](images/cooling_effect_map.png)

### SHAP Explainability

![SHAP Analysis](images/shap_analysis.png)

### What-if Predictor

![Predictor](images/what_if_predictor.png)

### Intervention Recommendations

![Recommendations](images/intervention_recommendations.png)

---

## 📖 Project Overview

This project presents a complete **AI/ML-powered Urban Heat Island (UHI) Decision Support System** developed for **Delhi NCR**.

Using satellite imagery, meteorological observations, urban morphology, and air-quality indicators, the system predicts **Land Surface Temperature (LST)**, identifies urban heat hotspots, explains the physical drivers of heating using SHAP, and recommends optimized cooling interventions for each hotspot grid.

The dashboard provides interactive visualizations that help researchers, planners, and policymakers explore heat stress patterns and evaluate mitigation strategies.

## 📌 What It Does
- Predicts Land Surface Temperature (LST) across 1,858 Delhi grids
- Classifies grids into Severe / High / Moderate heat stress zones
- Identifies 187 priority hotspot grids (top 10% LST)
- Quantifies 12 physical drivers via SHAP explainability
- Simulates 9 cooling intervention scenarios with per-grid °C reduction
- Generates optimal intervention strategy for urban planners

## 📊 Model Performance
| Metric | Value |
|--------|-------|
| Spatial CV R² | 0.9550 ± 0.0021 |
| Test R² | 0.9697 |
| Test RMSE | 1.32°C |
| Physics checks passed | 10/10 |

## 🛰️ Data Sources
| Source | Variables |
|--------|-----------|
| Landsat 8 | Land Surface Temperature |
| Sentinel-2 | NDVI, NDBI, MNDWI, Albedo |
| ERA5 / CPCB | Air temperature, humidity, wind speed |
| Sentinel-5P | NO₂, CO, SO₂ column density |
| GHSL / UT-GLOBUS | Built volume, impervious surface |

## 🏙️ Cooling Interventions Modelled
- 🌿 Urban Greening (NDVI +0.10) → ~1.5°C mean cooling
- 💧 Water Body Enhancement (MNDWI +0.10) → ~3.4°C mean cooling
- ☀️ Cool Roofs / Albedo increase (+0.10) → ~1.2°C mean cooling
- 🏗️ Built-up Surface Treatment (NDBI -0.05) → ~2.7°C mean cooling
- 🔗 Combined Aggressive scenario → maximum cooling

## 🚀 Run Locally
\`\`\`bash
pip install -r requirements.txt
streamlit run dashboard.py
\`\`\`

## 📁 Key Output Files
- `delhi_heat_stress_map.html` — interactive LST map (all grids)
- `delhi_hotspots_map.html` — top 10% hotspot grids
- `delhi_cooling_map.html` — per-grid cooling after intervention
- `optimal_intervention_strategy_v2.csv` — full strategy table

## ⚙️ Tech Stack
Python · XGBoost · SHAP · Folium · Streamlit · 
Pandas · Scikit-learn · Google Earth Engine (data collection)
