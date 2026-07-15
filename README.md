# Delhi-UHI-Dashboard
# 🌡️ Delhi Urban Heat Island — AI/ML Mitigation Dashboard

An end-to-end geospatial AI/ML system to identify urban heat stress 
hotspots, quantify key drivers of urban heating, and generate 
optimized scenario-based cooling interventions for Delhi NCT.

## 🔗 Live Dashboard
[delhi-uhi-dashboard.streamlit.app](https://delhi-uhi-dashboard.streamlit.app)

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
