import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delhi Urban Heat Island Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem; }

    .kpi-card {
        background: linear-gradient(135deg, #1e2130, #2a2f45);
        border: 1px solid #3a3f5c; border-radius: 12px;
        padding: 20px 24px; text-align: center;
    }
    .kpi-value { font-size: 2.2rem; font-weight: 700; color: #f0a500; margin: 0; }
    .kpi-label { font-size: 0.85rem; color: #9aa0b8; margin-top: 4px; }
    .kpi-sub   { font-size: 0.78rem; color: #6b7280; margin-top: 2px; }

    /* Page heading — bold, visible, always renders even if custom div fails */
    .page-heading {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #f5f7fa !important;
        border-left: 6px solid #f0a500;
        padding-left: 16px;
        margin-top: 0rem;
        margin-bottom: 1.2rem;
        line-height: 1.3;
    }
    .section-title {
        font-size: 1.4rem; font-weight: 600; color: #e2e8f0;
        border-left: 4px solid #f0a500; padding-left: 12px;
        margin-top: 0.4rem; margin-bottom: 1rem;
    }
    .badge-severe   { background:#7f1d1d; color:#fca5a5; padding:3px 10px; border-radius:20px; font-size:0.8rem; }
    .badge-high     { background:#7c2d12; color:#fdba74; padding:3px 10px; border-radius:20px; font-size:0.8rem; }
    .badge-moderate { background:#713f12; color:#fde68a; padding:3px 10px; border-radius:20px; font-size:0.8rem; }

    .technique-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #3a3f5c; border-left: 5px solid #f0a500;
        border-radius: 10px; padding: 18px 22px; margin-bottom: 14px;
    }
    .technique-title { font-size: 1.05rem; font-weight: 700; color: #f0a500; margin-bottom: 6px; }
    .technique-body  { font-size: 0.92rem; color: #cbd5e1; line-height: 1.55; }
    .technique-label { color: #94a3b8; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def page_heading(text: str):
    """Render a heading that is guaranteed to show up.
    Uses both a native Streamlit element (always renders) and a styled div."""
    st.title(text)


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("final_xgb_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_dataset():
    return pd.read_csv("Delhi_UHI_Final_Dataset_2019_2025.csv")

@st.cache_data
def load_interventions():
    return pd.read_csv("optimal_intervention_strategy_v2.csv")


@st.cache_data
def build_grid_data():
    """
    Load dataset and return a per-grid aggregated dataframe with zone labels.
    Returns (grid_df, p70, p90, error_message).
    error_message is None on success, otherwise a string explaining what failed
    (shown directly in the UI instead of being swallowed).
    """
    # Step 1 — load the raw dataset
    try:
        df = load_dataset()
    except FileNotFoundError:
        return None, None, None, "Dataset file `Delhi_UHI_Final_Dataset_2019_2025.csv` not found in the app folder."
    except Exception as e:
        return None, None, None, f"Could not read dataset CSV: {e}"

    # Step 2 — check required columns exist
    required = ["latitude", "longitude", "lst_celsius"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, None, None, f"Dataset is missing required column(s): {missing}. Found columns: {list(df.columns)}"

    # Step 3 — aggregate to grid level
    # NOTE: latitude/longitude are the groupby keys, so they must be excluded
    # from agg_cols — including them causes "cannot insert longitude, already exists"
    value_cols = [c for c in ["lst_celsius", "ndvi", "ndbi", "mndwi", "albedo",
                               "wind_speed", "humidity"]
                  if c in df.columns]
    if "lst_celsius" not in value_cols:
        return None, None, None, "Required column 'lst_celsius' not found in dataset."

    try:
        grid_df = (
            df.groupby(["latitude", "longitude"], as_index=False)[value_cols]
              .mean()
        )
    except Exception as e:
        return None, None, None, f"Failed to aggregate grid data: {e}"

    if grid_df.empty:
        return None, None, None, "Grid aggregation produced an empty table — check the dataset content."

    # Step 4 — zone classification
    p70 = grid_df["lst_celsius"].quantile(0.70)
    p90 = grid_df["lst_celsius"].quantile(0.90)

    def zone(v):
        if v >= p90:
            return "Severe"
        if v >= p70:
            return "High"
        return "Moderate"

    grid_df["zone"] = grid_df["lst_celsius"].apply(zone)

    # Step 5 — try to merge intervention data (optional, non-fatal if it fails)
    try:
        intv = load_interventions()
        if "latitude" in intv.columns and "longitude" in intv.columns:
            merge_cols = [c for c in intv.columns
                          if c not in grid_df.columns or c in ["latitude", "longitude"]]
            grid_df = grid_df.merge(intv[merge_cols], on=["latitude", "longitude"], how="left")
    except Exception:
        pass  # intervention file is optional for the LST/hotspot maps

    return grid_df, p70, p90, None


DELHI_CENTER = [28.6139, 77.2090]
ZONE_COLORS  = {"Severe": "#ef4444", "High": "#f97316", "Moderate": "#eab308"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌡️ Delhi UHI Dashboard")
    st.markdown("*Urban Heat Island Analysis · 2019–2025*")
    st.divider()
    screen = st.radio("Navigate", [
        "📊 Overview",
        "🗺️ Heat Stress Map",
        "🔍 SHAP Feature Importance",
        "🎛️ What-If Predictor",
        "🏙️ Intervention Recommendations",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("**Model:** XGBoost (Spatial CV)")
    st.markdown("**R²:** 0.9550 &nbsp;|&nbsp; **RMSE:** 1.58°C", unsafe_allow_html=True)
    st.markdown("**Data:** 140,585 grid observations")

# ═══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if screen == "📊 Overview":
    page_heading("📊 Project Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (val, label, sub) in zip([c1, c2, c3, c4, c5], [
        ("0.9550", "Model R²", "XGBoost Spatial CV"),
        ("1.58°C", "RMSE", "Test set error"),
        ("140,585", "Grid Observations", "2019 – 2025"),
        ("12", "Features Used", "Satellite + Weather"),
        ("9/10", "Physics Checks", "Model validated ✅"),
    ]):
        with col:
            st.markdown(f'<div class="kpi-card"><p class="kpi-value">{val}</p>'
                        f'<p class="kpi-label">{label}</p><p class="kpi-sub">{sub}</p></div>',
                        unsafe_allow_html=True)
    st.divider()
    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### 🏆 Model Comparison")
        st.dataframe(pd.DataFrame({
            "Metric":        ["R² (mean)", "RMSE (mean)", "MAE (mean)"],
            "XGBoost":       [0.9550, 1.5838, 1.0390],
            "Random Forest": [0.8687, 2.7071, 1.9076],
            "Winner":        ["✅ XGBoost", "✅ XGBoost", "✅ XGBoost"],
        }), use_container_width=True, hide_index=True)
        st.markdown("#### 🌡️ Heat Stress Zones")
        st.markdown("""
| Zone | Threshold | Meaning |
|------|-----------|---------|
| 🔴 Severe   | Top 10% LST    | Urgent intervention |
| 🟠 High     | Top 10–30% LST | Monitor + plan      |
| 🟡 Moderate | Bottom 70% LST | Baseline            |
""")
    with cr:
        st.markdown("#### 🔬 Physics Validation (9/10 passed)")
        st.dataframe(pd.DataFrame({
            "Feature":   ["albedo", "ndbi", "ndvi", "mndwi", "air_temp_celsius",
                          "humidity", "wind_speed", "isa_percent", "built_volume_total", "co_column_density"],
            "Direction": ["↓ Cooling", "↑ Heating", "↓ Cooling", "↓ Cooling", "↑ Heating",
                          "↓ Cooling", "↓ Cooling", "↑ Heating", "↑ Heating", "↑ Heating"],
            "Result":    ["✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "❌"],
            "r":         [-0.161, +0.939, -0.882, -0.941, +0.962, -0.833, -0.335, +0.670, +0.792, -0.183],
        }), use_container_width=True, hide_index=True)
        st.markdown("#### 📋 Top Drivers")
        st.dataframe(pd.DataFrame({
            "Rank":         range(1, 6),
            "Feature":      ["air_temp_celsius", "ndbi", "humidity", "mndwi", "wind_speed"],
            "Contribution": ["40.3%", "17.5%", "11.3%", "8.9%", "7.1%"],
            "Role":         ["↑ Heating", "↑ Heating", "↓ Cooling", "↓ Cooling", "↓ Cooling"],
        }), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — HEAT STRESS MAPS (3 sub-maps)
# ═══════════════════════════════════════════════════════════════════════════════
elif screen == "🗺️ Heat Stress Map":
    page_heading("🗺️ Delhi Heat Stress Maps")

    tab1, tab2, tab3 = st.tabs([
        "🌐 General LST Map",
        "🔴 Hotspot Map",
        "🌿 Cooling Effect Map (with Intervention Strategy)",
    ])

    # ── TAB 1: General LST Map ──────────────────────────────────────────────
    with tab1:
        st.markdown("Shows XGBoost-predicted LST across all Delhi grids. Click any grid for details.")
        map_file = "delhi_heat_stress_map.html"
        if os.path.exists(map_file):
            with open(map_file, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=580, scrolling=False)
        else:
            st.error(f"`{map_file}` not found in the app folder. "
                     f"Copy it from your notebook output into the same folder as dashboard.py.",
                     icon="🚨")

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Severe",   "Top 10% LST",    "Urgent intervention")
        c2.metric("🟠 High",     "Top 10–30% LST", "Monitor + plan")
        c3.metric("🟡 Moderate", "Bottom 70% LST", "Baseline")

    # ── TAB 2: Hotspot Map ──────────────────────────────────────────────────
    with tab2:
        st.markdown("Shows only the **top 10% LST grids** — the 187 urban heat hotspots. "
                    "Click any point for its exact LST value.")
        map_file = "delhi_hotspots_map.html"
        if os.path.exists(map_file):
            with open(map_file, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=580, scrolling=False)
        else:
            st.error(f"`{map_file}` not found in the app folder. "
                     f"Copy it from your notebook output into the same folder as dashboard.py.",
                     icon="🚨")

    # ── TAB 3: Cooling Effect Map (with full intervention strategy) ─────────
    with tab3:
        st.markdown(
            "Shows the **187 hotspot grids** coloured by cooling achieved under the best "
            "scenario. **Click any coloured point** — the popup shows the exact baseline LST, "
            "cooling achieved, post-intervention LST, and the recommended intervention for "
            "that specific grid."
        )
        map_file = "delhi_cooling_map.html"
        if os.path.exists(map_file):
            with open(map_file, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600, scrolling=False)

            st.divider()
            st.markdown("##### 📊 What the popup numbers mean")
            st.markdown("""
| Value shown on click | Meaning |
|---|---|
| **Priority Rank** | This grid's rank among all 187 hotspots by baseline LST (1 = hottest) |
| **🌡 Baseline LST** | Predicted LST *before* any intervention |
| **❄ Cooling Achieved** | Exact °C reduction from applying the recommended intervention |
| **✅ Post-intervention LST** | Baseline LST minus cooling — the expected result after action |
| **Recommended Action** | The specific intervention type for that grid (Greening / Water / Cool Roofs / Built-up Treatment) |
""")

            try:
                strategy_df = pd.read_csv("optimal_intervention_strategy_v2.csv")
                cool_col = next((c for c in strategy_df.columns
                                 if any(k in c.lower() for k in ["cool", "reduc"])), None)
                lst_col  = next((c for c in strategy_df.columns
                                 if ("base" in c.lower() or "predicted" in c.lower()) and "lst" in c.lower()), None)
                post_col = next((c for c in strategy_df.columns if "post" in c.lower() and "lst" in c.lower()), None)
                rec_col  = next((c for c in strategy_df.columns if "recommend" in c.lower()), None)
                feas_col = next((c for c in strategy_df.columns if "feasib" in c.lower()), None)
                rank_col = next((c for c in strategy_df.columns if "rank" in c.lower()), None)
                grid_col = next((c for c in strategy_df.columns if "grid" in c.lower() and "id" in c.lower()), None)

                if cool_col:
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Hotspot grids",         f"{len(strategy_df):,}")
                    sc2.metric("Avg cooling achieved",  f"{strategy_df[cool_col].mean():.2f}°C")
                    sc3.metric("Max cooling achieved",  f"{strategy_df[cool_col].max():.2f}°C")
                    if lst_col:
                        sc4.metric("Avg baseline LST",  f"{strategy_df[lst_col].mean():.2f}°C")

                st.divider()
                vc1, vc2 = st.columns([1, 1])

                with vc1:
                    st.markdown("##### 🔥 Top 10 Priority Hotspots")
                    show_cols = [c for c in [rank_col, grid_col, lst_col, cool_col, post_col, rec_col]
                                if c is not None]
                    if rank_col:
                        top10 = strategy_df.sort_values(rank_col).head(10)
                    elif lst_col:
                        top10 = strategy_df.sort_values(lst_col, ascending=False).head(10)
                    else:
                        top10 = strategy_df.head(10)
                    st.dataframe(top10[show_cols], use_container_width=True, hide_index=True, height=300)

                with vc2:
                    if rec_col:
                        st.markdown("##### 📋 Intervention Type Distribution")
                        dist = strategy_df[rec_col].value_counts().reset_index()
                        dist.columns = ["Recommendation", "Grid Count"]
                        dist["Share"] = (dist["Grid Count"] / dist["Grid Count"].sum() * 100).round(1).astype(str) + "%"
                        st.dataframe(dist, use_container_width=True, hide_index=True, height=300)

                if feas_col:
                    st.markdown("##### ✅ Feasibility Breakdown")
                    feas_dist = strategy_df[feas_col].value_counts()
                    fc1, fc2, fc3 = st.columns(3)
                    for col, level in zip([fc1, fc2, fc3], ["High", "Medium", "Low"]):
                        count = feas_dist.get(level, 0)
                        col.metric(f"{level} feasibility", f"{count:,} grids")

            except FileNotFoundError:
                st.info("`optimal_intervention_strategy_v2.csv` not found in the app folder — "
                        "place it alongside dashboard.py to show the strategy table here.", icon="ℹ️")
            except Exception as e:
                st.warning(f"Could not load intervention strategy values: {e}", icon="⚠️")
        else:
            st.error(f"`{map_file}` not found in the app folder. "
                     f"Copy it from your notebook output into the same folder as dashboard.py.",
                     icon="🚨")

# ═══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — SHAP FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif screen == "🔍 SHAP Feature Importance":
    page_heading("🔍 SHAP Feature Importance — What Drives LST?")

    shap_bar  = "SHAP_Bar.png"
    shap_full = "SHAP_Summary.png"

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 📊 Mean |SHAP| Bar Chart")
        if os.path.exists(shap_bar):
            st.image(shap_bar, use_container_width=True,
                     caption="Mean absolute SHAP values — higher = stronger influence on LST")
        else:
            st.warning(f"`{shap_bar}` not found.", icon="⚠️")

    with col_b:
        st.markdown("##### 🐝 SHAP Summary / Beeswarm Plot")
        if os.path.exists(shap_full):
            st.image(shap_full, use_container_width=True,
                     caption="SHAP summary plot — direction and spread of each feature's impact")
        else:
            st.info("Place `SHAP_Summary.png` (your beeswarm/dot plot) here to display it.", icon="ℹ️")

    st.divider()
    st.markdown("#### 📋 Driver Summary Table")
    shap_data = pd.DataFrame({
        "Rank":             range(1, 11),
        "Feature":          ["air_temp_celsius", "ndbi", "humidity", "mndwi", "wind_speed",
                             "albedo", "built_volume_total", "ndvi", "co_column_density", "isa_percent"],
        "Mean |SHAP|":      [3.76, 1.63, 1.05, 0.83, 0.66, 0.27, 0.23, 0.21, 0.20, 0.09],
        "% Contribution":   ["40.3%", "17.5%", "11.3%", "8.9%", "7.1%", "2.9%", "2.4%", "2.3%", "2.2%", "1.0%"],
        "Direction":        ["↑ Heating", "↑ Heating", "↓ Cooling", "↓ Cooling", "↓ Cooling",
                             "↓ Cooling", "↑ Heating", "↓ Cooling", "↑ Heating", "↑ Heating"],
        "Physical Role":    [
            "Atmospheric heating — dominant driver",
            "Built-up surfaces absorb solar radiation",
            "Evapotranspiration — latent cooling",
            "Water bodies cool via evaporation",
            "Advective heat dispersal",
            "Reflective surfaces reduce solar absorption",
            "Urban canyon traps heat",
            "Vegetation cools via evapotranspiration",
            "Combustion — anthropogenic heat source",
            "Impervious surfaces retain ground heat",
        ],
    })
    st.dataframe(shap_data, use_container_width=True, hide_index=True)
    st.caption("↑ Heating = raises LST · ↓ Cooling = lowers LST")

# ═══════════════════════════════════════════════════════════════════════════════
# SCREEN 4 — WHAT-IF PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif screen == "🎛️ What-If Predictor":
    page_heading("🎛️ What-If LST Predictor")
    st.markdown("Adjust sliders **or search by lat/lon** to get a live XGBoost LST prediction.")

    model_loaded = False
    try:
        model = load_model()
        model_loaded = True
    except FileNotFoundError:
        st.error("`final_xgb_model.pkl` not found.", icon="🚨")

    with st.expander("🔍 Search by Latitude / Longitude (auto-fills sliders)", expanded=False):
        sc1, sc2, sc3 = st.columns([2, 2, 1])
        search_lat = sc1.number_input("Latitude",  value=28.6139, format="%.4f")
        search_lon = sc2.number_input("Longitude", value=77.2090, format="%.4f")
        search_btn = sc3.button("Find nearest grid", use_container_width=True)

        default_vals = {}
        if search_btn:
            try:
                df = load_dataset()
                df["_dist"] = ((df["latitude"] - search_lat) ** 2 + (df["longitude"] - search_lon) ** 2) ** 0.5
                nearest = df.loc[df["_dist"].idxmin()]
                st.success(f"Nearest grid: ({nearest['latitude']:.4f}, {nearest['longitude']:.4f})"
                           f" — LST: {nearest['lst_celsius']:.1f}°C")
                default_vals = nearest.to_dict()
                st.session_state["prefill"] = default_vals
            except Exception as e:
                st.warning(f"Could not search dataset: {e}", icon="⚠️")

    pf = st.session_state.get("prefill", {})

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("##### 🌤️ Meteorological")
        air_temp   = st.slider("Air Temperature (°C)",      15.0, 50.0, float(pf.get("air_temp_celsius", 35.0)), 0.5)
        humidity   = st.slider("Humidity (%)",              10.0, 95.0, float(pf.get("humidity", 40.0)),          1.0)
        wind_speed = st.slider("Wind Speed (m/s)",          0.0,  10.0, float(pf.get("wind_speed", 2.5)),         0.1)

        st.markdown("##### 🛰️ Remote Sensing")
        ndvi   = st.slider("NDVI (Vegetation)",  -0.2, 0.8,  float(pf.get("ndvi",  0.15)), 0.01)
        ndbi   = st.slider("NDBI (Built-up)",    -0.5, 0.5,  float(pf.get("ndbi",  0.02)), 0.01)
        mndwi  = st.slider("MNDWI (Water)",      -0.6, 0.4,  float(pf.get("mndwi", -0.19)), 0.01)
        albedo = st.slider("Albedo",              0.05, 0.40, float(pf.get("albedo", 0.14)), 0.01)

    with col_r:
        st.markdown("##### 🏗️ Urban Morphology")
        isa_percent  = st.slider("Impervious Surface (%)",    0.0, 100.0, float(pf.get("isa_percent", 60.0)),          1.0)
        built_volume = st.slider("Built Volume (m³/km²)",     0.0, 5000.0, float(pf.get("built_volume_total", 1200.0)), 50.0)
        population   = st.slider("Population Count",          0,   50000,  int(pf.get("population_count", 8000)),      500)

        st.markdown("##### 💨 Air Quality")
        co_density = st.slider("CO Column Density (mol/m²)", 0.01, 0.10, float(pf.get("co_column_density", 0.04)), 0.001, format="%.3f")
        no2        = st.slider("Tropospheric NO₂ (mol/m²)", 0.0, 0.0002, float(pf.get("tropospheric_no2", 0.00008)), 0.000005, format="%.6f")

        st.divider()
        if model_loaded:
            features_arr = np.array([[ndvi, ndbi, mndwi, air_temp, humidity, wind_speed,
                                       built_volume, albedo, co_density, isa_percent,
                                       population, no2]])
            try:
                pred_lst = model.predict(features_arr)[0]
            except Exception:
                features_arr = np.array([[ndvi, ndbi, mndwi, air_temp, humidity, wind_speed,
                                           built_volume, albedo, co_density, isa_percent]])
                pred_lst = model.predict(features_arr)[0]

            if pred_lst >= 45:
                zone, color, badge = "🔴 SEVERE", "#ef4444", "badge-severe"
            elif pred_lst >= 43:
                zone, color, badge = "🟠 HIGH", "#f97316", "badge-high"
            else:
                zone, color, badge = "🟡 MODERATE", "#eab308", "badge-moderate"

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1e2130,#2a2f45);border:1px solid #3a3f5c;
                        border-radius:14px;padding:28px;text-align:center;">
                <p style="color:#9aa0b8;font-size:0.9rem;margin:0">XGBoost Predicted LST</p>
                <p style="color:{color};font-size:3.5rem;font-weight:800;margin:8px 0">{pred_lst:.1f}°C</p>
                <span class="{badge}">{zone}</span>
                <hr style="border-color:#3a3f5c;margin:16px 0">
                <p style="color:#9aa0b8;font-size:0.8rem;margin:0">R² = 0.9550 · RMSE = 1.58°C</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("##### 💡 Cooling Suggestions")
            tips = []
            if ndvi < 0.15:       tips.append("🌿 Increase vegetation (NDVI low)")
            if ndbi > 0.1:        tips.append("🏗 Add green roofs / reflective coatings (NDBI high)")
            if albedo < 0.15:     tips.append("☀ Apply cool-roof paint (albedo low)")
            if isa_percent > 70:  tips.append("🧱 Use permeable pavements (ISA very high)")
            if humidity < 30:     tips.append("💧 Add water features (humidity low)")
            if not tips:          tips.append("✅ Conditions balanced for current LST")
            for t in tips:
                st.markdown(f"- {t}")
        else:
            st.info("Place `final_xgb_model.pkl` to enable predictions.", icon="ℹ️")

# ═══════════════════════════════════════════════════════════════════════════════
# SCREEN 5 — INTERVENTION RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif screen == "🏙️ Intervention Recommendations":
    page_heading("🏙️ Intervention Recommendations")
    st.caption(
        "📌 Decision-support output for urban planners — prioritizes *where* to act and "
        "*what type* of intervention to investigate. Not a final construction or engineering plan."
    )

    csv_file = "optimal_intervention_strategy_v2.csv"
    if not os.path.exists(csv_file):
        st.error(f"`{csv_file}` not found.", icon="🚨")
    else:
        df = load_interventions()
        total = len(df)

        zone_col = next((c for c in df.columns if "zone" in c.lower() or "stress" in c.lower()), None)
        lst_col  = next((c for c in df.columns if "lst" in c.lower()), None)
        cool_col = next((c for c in df.columns if any(k in c.lower() for k in ["cool", "reduc", "saving", "delta"])), None)
        lat_col  = next((c for c in df.columns if "lat" in c.lower()), None)
        lon_col  = next((c for c in df.columns if "lon" in c.lower()), None)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Grids", f"{total:,}")
        if zone_col:
            sev = df[zone_col].astype(str).str.contains("evere|HIGH|high", na=False).sum()
            c2.metric("High Priority Grids", f"{sev:,}", f"{sev/total*100:.1f}%")
        if lst_col:
            c3.metric("Avg Baseline LST", f"{df[lst_col].mean():.1f}°C")
        if cool_col:
            c4.metric("Avg Cooling Potential", f"{df[cool_col].mean():.2f}°C")

        st.divider()

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            if zone_col:
                zones = ["All"] + sorted(df[zone_col].dropna().unique().tolist())
                sel_zone = st.selectbox("Filter by Zone", zones)
            else:
                sel_zone = "All"
        with fc2:
            grid_search = st.text_input("Search grid ID or any value", "")
        with fc3:
            st.markdown("**Or search by Lat / Lon:**")
            ll1, ll2 = st.columns(2)
            srch_lat = ll1.number_input("Lat", value=0.0, format="%.4f", label_visibility="collapsed")
            srch_lon = ll2.number_input("Lon", value=0.0, format="%.4f", label_visibility="collapsed")
            ll_btn   = st.button("Find nearest", use_container_width=True)

        filtered = df.copy()
        if sel_zone != "All" and zone_col:
            filtered = filtered[filtered[zone_col] == sel_zone]
        if grid_search:
            mask = filtered.astype(str).apply(lambda col: col.str.contains(grid_search, case=False)).any(axis=1)
            filtered = filtered[mask]
        if ll_btn and lat_col and lon_col and (srch_lat != 0.0 or srch_lon != 0.0):
            filtered["_dist"] = ((df[lat_col] - srch_lat) ** 2 + (df[lon_col] - srch_lon) ** 2) ** 0.5
            nearest_idx = filtered["_dist"].idxmin()
            nearest_row = filtered.loc[nearest_idx]
            st.success(f"Nearest grid: lat={nearest_row[lat_col]:.4f}, lon={nearest_row[lon_col]:.4f}")
            filtered = filtered.loc[[nearest_idx]].drop(columns=["_dist"], errors="ignore")

        st.markdown(f"Showing **{len(filtered):,}** of **{total:,}** rows")
        st.dataframe(filtered, use_container_width=True, height=380)

        csv_bytes = filtered.to_csv(index=False).encode()
        st.download_button("⬇️ Download filtered CSV", csv_bytes,
                           file_name="filtered_interventions.csv", mime="text/csv")

        st.info(
            "**How to read this table — decision-support, not a construction plan.** "
            "This output is meant to **prioritize and guide** urban planners, not replace "
            "site-level engineering design.\n\n"
            "- **Priority rank / baseline LST** — tells planners *where* to focus first, "
            "based on which 1km grids are hottest.\n"
            "- **Recommendation** — tells planners *what category* of intervention is "
            "physically justified for that grid (derived from its NDVI/NDBI/MNDWI/albedo "
            "values relative to the city average) — not the exact street, plot, or building.\n"
            "- **Cooling (°C)** — is the model's **estimated** temperature reduction if the "
            "intervention is fully applied across that grid. It is a projection from the "
            "trained ML model, not a guaranteed or measured real-world outcome.\n"
            "- **Feasibility** — is a rough High/Medium/Low rating of how disruptive that "
            "intervention category typically is, not a cost or permissions assessment.\n\n"
            "Before implementation, planners would still need ground-level site surveys, "
            "feasibility studies, land-ownership checks, and detailed engineering design. "
            "This tool's role is to make that downstream process faster and more targeted "
            "by narrowing down *where* and *what kind* of intervention to investigate first.",
            icon="ℹ️",
        )

        st.divider()

        # ── Intervention techniques — how each works + what to do ────────────
        st.markdown("#### 🛠️ Intervention Techniques — How They Work & What To Do")
        st.markdown(
            "Each recommended action below targets a specific physical driver of heat "
            "identified by the model (NDVI, NDBI, Albedo, MNDWI). Use these as implementation "
            "guidance once a grid has been flagged for that intervention."
        )

        techniques = [
            {
                "title": "🌿 Urban Greening (raises NDVI)",
                "trigger": "Recommended when a grid's NDVI is below the city average — i.e. low vegetation cover.",
                "how_it_works": (
                    "Vegetation cools the surface mainly through evapotranspiration — water "
                    "evaporating from leaves absorbs latent heat from the surrounding air, directly "
                    "lowering local LST. Tree canopies also provide shade that blocks direct solar "
                    "radiation from reaching paved/built surfaces."
                ),
                "what_to_do": (
                    "Plant street trees along roads and footpaths; convert unused plots into pocket "
                    "parks; introduce rooftop and terrace gardens on government and commercial "
                    "buildings; prioritise native, low-water species suited to Delhi's climate to "
                    "keep maintenance and irrigation costs low."
                ),
            },
            {
                "title": "💧 Water Body Restoration (raises MNDWI)",
                "trigger": "Recommended when a grid's MNDWI is below the city average — i.e. little to no nearby water surface.",
                "how_it_works": (
                    "Open water bodies cool the surrounding air through evaporation, and they also "
                    "have high thermal mass, meaning they heat up and cool down more slowly than "
                    "concrete or asphalt — reducing peak daytime LST in their vicinity."
                ),
                "what_to_do": (
                    "Restore and de-silt existing stormwater drains and canals; rejuvenate neighbourhood "
                    "ponds and lakes (e.g. along Yamuna floodplain restoration efforts already underway in Delhi); "
                    "introduce small constructed wetlands or retention ponds within new developments; "
                    "ensure water features are functional year-round, not just during monsoon."
                ),
            },
            {
                "title": "☀ Cool Roofs & Reflective Surfaces (raises Albedo)",
                "trigger": "Recommended when a grid's Albedo is below the city average — i.e. dark, heat-absorbing surfaces dominate.",
                "how_it_works": (
                    "Albedo measures how much solar radiation a surface reflects versus absorbs. Dark "
                    "roofs and asphalt absorb most incoming sunlight and re-radiate it as heat. Raising "
                    "albedo means more sunlight is reflected away before it can heat the surface."
                ),
                "what_to_do": (
                    "Apply white or light-coloured reflective paint/coating on flat rooftops; use "
                    "high-albedo (light-coloured) materials for new pavements and parking lots; "
                    "retrofit existing dark asphalt roads in severe-zone grids with reflective sealants "
                    "during routine resurfacing cycles."
                ),
            },
            {
                "title": "🏗 Built-up Surface Treatment (lowers NDBI)",
                "trigger": "Recommended when a grid's NDBI is above the city average — i.e. dense built-up/impervious cover.",
                "how_it_works": (
                    "NDBI reflects how much of the surface is impervious built-up material, which absorbs "
                    "heat during the day and releases it slowly at night, keeping LST elevated. This does "
                    "NOT require demolishing buildings — only changing the surface properties and density "
                    "of impervious cover at street level."
                ),
                "what_to_do": (
                    "Replace solid concrete pavements with permeable, porous paving that allows water "
                    "infiltration and reduces heat retention; install vertical green walls on building "
                    "facades to reduce exposed wall thermal mass; add shaded walkways or pergolas over "
                    "pedestrian areas; convert flat concrete rooftops into rooftop gardens — all of which "
                    "lower the effective heat-absorption signature without structural demolition."
                ),
            },
        ]

        for t in techniques:
            st.markdown(f"""
            <div class="technique-card">
                <div class="technique-title">{t['title']}</div>
                <div class="technique-body">
                    <span class="technique-label">When recommended:</span> {t['trigger']}<br><br>
                    <span class="technique-label">How it works:</span> {t['how_it_works']}<br><br>
                    <span class="technique-label">What to do:</span> {t['what_to_do']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 📋 Master Plan 2041 — Alignment Summary")
        st.markdown("""
| Intervention | Mechanism | Impact |
|---|---|---|
| **Permeable pavements** | Water infiltration → reduces surface heat retention | ↓ NDBI, ↓ ISA |
| **Vertical green walls** | Reduces thermal mass of building facades | ↓ NDBI, ↓ LST |
| **Reflective road coatings** | Raises albedo, reduces solar absorption | ↑ Albedo |
| **Shaded walkways / pergolas** | Overhead shade blocks direct radiation | ↓ LST locally |
| **Rooftop gardens** | Converts concrete rooftops to planted surfaces | ↑ NDVI, ↓ NDBI |

All interventions are implementable within existing **Delhi Master Plan 2041** frameworks.
""")
