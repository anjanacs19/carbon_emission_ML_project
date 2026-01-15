import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="CarbonSight", layout="wide")

st.markdown("""
<style>
header, footer, #MainMenu {visibility: hidden;}
.stApp {background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #1a1a2e, #16213e);}
h1, h2, h3 {color: white !important;}


    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff5252 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 30px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        width: auto !important;
        min-width: 200px !important;
        transition: all 0.3s !important;
    }
    
    .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #ff5252 0%, #ff3838 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(255, 107, 107, 0.3) !important;
    }



/* Base sidebar button style */
.sidebar-btn {
    padding: 10px;
    border-radius: 6px;
    text-align: center;
    margin: 5px 0;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid rgba(255,255,255,0.2);
    color: #dddddd;
    background: transparent;
}

/* Active (current page) button */
.sidebar-btn.active {
    background: linear-gradient(135deg, #ff6b6b 0%, #ff5252 100%);
    border: 1px solid #ff6b6b;
    color: white;
}


    
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA / MODEL / ENCODERS
# -------------------------------------------------
@st.cache_resource
def load_assets():
    model = pickle.load(open("carbon_model.pkl", "rb"))
    sector_encoder = pickle.load(open("sector_encoder.pkl", "rb"))
    transport_encoder = pickle.load(open("transport_encoder.pkl", "rb"))
    df = pd.read_csv("carbon_emission_dataset.csv")
    return model, sector_encoder, transport_encoder, df

model, sector_encoder, transport_encoder, df = load_assets()

industry_sectors = list(sector_encoder.classes_)
transport_modes = list(transport_encoder.classes_)

X_COLUMNS = [
 'Sector',
 'Total_Energy_Consumption_kWh',
 'Renewable_Energy_Consumption_kWh',
 'NonRenewable_Energy_Consumption_kWh',
 'Production_Output_Units',
 'Supply_Chain_Transport_km',
 'Supply_Chain_Transport_Mode',
 'Raw_Material_Usage_kg',
 'Energy_Cost_USD',
 'Carbon_Tax_USD',
 'Process_Efficiency_Percent'
]

# # -------------------------------------------------
# # SIDEBAR NAV
# # -------------------------------------------------
# if "page" not in st.session_state:
#     st.session_state.page = "prediction"

# st.sidebar.markdown("---")
# if st.sidebar.button("💡 Current Emission", use_container_width=True):
#     st.session_state.page = "prediction"
# if st.sidebar.button("🎯 Scenarios", use_container_width=True):
#     st.session_state.page = "target"
# st.sidebar.markdown("---")


# -------------------------------------------------
# SIDEBAR NAV
# -------------------------------------------------
# if "page" not in st.session_state:
#     st.session_state.page = "prediction"

# st.sidebar.markdown("---")

# # Current Emission button
# if st.session_state.page == 'prediction':
#     st.sidebar.markdown("""
#     <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ff5252 100%); 
#     padding: 10px; border-radius: 6px; text-align: center; margin: 5px 0;
#     border: 1px solid #ff6b6b;'>
#         <span style='color: white; font-weight: 600;'>💡 Current Emission</span>
#     </div>
#     """, unsafe_allow_html=True)
# else:
#     if st.sidebar.button("💡 Current Emission", use_container_width=True):
#         st.session_state.page = "prediction"
#         st.rerun()

# # Scenarios button
# if st.session_state.page == 'target':
#     st.sidebar.markdown("""
#     <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ff5252 100%); 
#     padding: 10px; border-radius: 6px; text-align: center; margin: 5px 0;
#     border: 1px solid #ff6b6b;'>
#         <span style='color: white; font-weight: 600;'>🎯 Scenarios</span>
#     </div>
#     """, unsafe_allow_html=True)
# else:
#     if st.sidebar.button("🎯 Scenarios", use_container_width=True):
#         st.session_state.page = "target"
#         st.rerun()

# st.sidebar.markdown("---")

# -------------------------------------------------
# SIDEBAR NAV
# -------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "prediction"

st.sidebar.markdown("---")

# Current Emission
if st.session_state.page == "prediction":
    st.sidebar.markdown(
        "<div class='sidebar-btn active'>💡 Current Emission</div>",
        unsafe_allow_html=True
    )
else:
    if st.sidebar.button("💡 Current Emission", use_container_width=True):
        st.session_state.page = "prediction"
        st.rerun()

# Scenarios
if st.session_state.page == "target":
    st.sidebar.markdown(
        "<div class='sidebar-btn active'>🎯 Scenarios</div>",
        unsafe_allow_html=True
    )
else:
    if st.sidebar.button("🎯 Scenarios", use_container_width=True):
        st.session_state.page = "target"
        st.rerun()

st.sidebar.markdown("---")


# -------------------------------------------------
# PAGE 1 — PREDICTION
# -------------------------------------------------
if st.session_state.page == "prediction":

    st.markdown("<h1 style='text-align:center;'>🌍 Carbon Emission Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#aaa;'>Predict emissions based on your current operations</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("prediction_form"):

        c1, c2 = st.columns(2)
        with c1:
            sector = st.selectbox("🏭 Industry Sector", industry_sectors)
        with c2:
            transport = st.selectbox("🚚 Transport Mode", transport_modes)

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            total_energy = st.number_input("⚡ Total Energy (kWh)", value=152532.15)
        with c2:
            renewable_energy = st.number_input("🌱 Renewable Energy (kWh)", value=20698.80)

        non_renewable_energy = total_energy - renewable_energy

        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        with c1:
            production_output = st.number_input("📦 Production Output", value=6714.80)
        with c2:
            transport_km = st.number_input("🛣️ Transport Distance (km)", value=2903.63)
        with c3:
            efficiency = st.slider("⚙️ Process Efficiency (%)", 0.0, 100.0, 79.49)

        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        with c1:
            raw_material = st.number_input("🪨 Raw Material (kg)", value=21438.17)
        with c2:
            energy_cost = st.number_input("💵 Energy Cost (USD)", value=18192.00)
        with c3:
            carbon_tax = st.number_input("💰 Carbon Tax (USD)", value=1942.56)
        
        # st.markdown("---")

        
 
        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            submit = st.form_submit_button("🔮 Predict Emission", use_container_width=True, type="primary")
        # submit = st.form_submit_button("🔮 Predict Emission")

    if submit:

        # ENCODE EXACTLY LIKE TRAINING
        sector_enc = sector_encoder.transform([sector])[0]
        transport_enc = transport_encoder.transform([transport])[0]

        input_dict = {
            'Sector': sector_enc,
            'Total_Energy_Consumption_kWh': total_energy,
            'Renewable_Energy_Consumption_kWh': renewable_energy,
            'NonRenewable_Energy_Consumption_kWh': non_renewable_energy,
            'Production_Output_Units': production_output,
            'Supply_Chain_Transport_km': transport_km,
            'Supply_Chain_Transport_Mode': transport_enc,
            'Raw_Material_Usage_kg': raw_material,
            'Energy_Cost_USD': energy_cost,
            'Carbon_Tax_USD': carbon_tax,
            'Process_Efficiency_Percent': efficiency
        }

        X_input = pd.DataFrame([input_dict], columns=X_COLUMNS)

        prediction = model.predict(X_input)[0]

        st.markdown("---")
        # st.metric("🏭 Predicted Emission", f"{prediction:.2f} tCO₂e")
        
        
        
        # After prediction is made
        renewable_pct = (renewable_energy / total_energy * 100) if total_energy > 0 else 0

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: left; margin-left: 20px;'>📊 Prediction Results</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div style='margin: 10px;'>
                <p style='color: #a0b0c0; font-size: 18px; margin-bottom: 8px;'>📊 Predicted Emission</p>
                <p style='color: white; font-size: 36px; font-weight: bold; margin: 0;'>{prediction:.2f} tCO₂e</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='margin: 10px;'>
                <p style='color: #a0b0c0; font-size: 18px; margin-bottom: 8px;'>🌱 Renewable Share</p>
                <p style='color: white; font-size: 36px; font-weight: bold; margin: 0;'>{renewable_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # Add the success/info box below
        st.markdown("<br>", unsafe_allow_html=True)

        if prediction < 20:
            st.markdown("""
            <div style='background: rgba(76, 175, 80, 0.2); border: 1px solid rgba(76, 175, 80, 0.4); 
            border-radius: 10px; padding: 20px; margin: 20px;'>
                <p style='color: #81c784; margin: 0; font-size: 16px;'>
                    ✅ <strong>Low Emission Level</strong> 🟢 - Excellent environmental performance!
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif prediction < 50:
            st.markdown("""
            <div style='background: rgba(33, 150, 243, 0.2); border: 1px solid rgba(33, 150, 243, 0.4); 
            border-radius: 10px; padding: 20px; margin: 20px;'>
                <p style='color: #64b5f6; margin: 0; font-size: 16px;'>
                    ℹ️ <strong>Moderate Emission Level</strong> - Good performance with room for improvement.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: rgba(255, 152, 0, 0.2); border: 1px solid rgba(255, 152, 0, 0.4); 
            border-radius: 10px; padding: 20px; margin: 20px;'>
                <p style='color: #ffb74d; margin: 0; font-size: 16px;'>
                    ⚠️ <strong>High Emission Level</strong> - Consider implementing reduction strategies.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state["baseline_emission"] = prediction
            st.session_state["renewable_share"] = renewable_pct





# -------------------------------------------------
# PAGE 2 — SCENARIO ANALYSIS
# -------------------------------------------------
elif st.session_state.page == "target":

    st.markdown(
        "<h1 style='text-align:center;'>🎯 Target & Scenario Analysis</h1>",
        unsafe_allow_html=True
    )
    # st.markdown("<p style='color: #a0b0c0; font-size: 16px;'>Analyze sustainability targets and compare strategies</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#aaa;'>Analyze sustainability targets and compare strategies</p>", unsafe_allow_html=True)
    if "baseline_emission" not in st.session_state:
        st.warning("⚠️ Please run **Current Emission Prediction** first.")
        st.stop()

    baseline = st.session_state["baseline_emission"]
    renewable_percent = st.session_state["renewable_share"]

    st.markdown("### 📍 Current Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Current Emission", f"{baseline:.2f} tCO₂e")
    with c2:
        st.metric("Renewable Share", f"{renewable_percent:.1f}")

    st.markdown("---")

    mode = st.radio(
        "Choose Analysis Mode",
        ["🎯 Single Target Analysis", "🔮 Multiple Scenario Comparison"],
        horizontal=True
    )

    # =================================================
    # MODE 1 — SINGLE TARGET
    # =================================================
    if mode == "🎯 Single Target Analysis":

        st.markdown("### 🎯 Define Your Target")

        c1, c2 = st.columns(2)
        with c1:
            reduction_pct = st.slider(
                "Expected Emission Reduction (%)",
                0.0, 60.0, 20.0
            )
        with c2:
            renewable_gain = st.slider(
                "Additional Renewable Share (%)",
                0.0, 50.0, 10.0
            )

        if st.button("📊 Analyze Target", type="primary"):

            reduction_factor = reduction_pct / 100
            renewable_factor = renewable_gain / 200  # softer impact

            target_emission = baseline * max(
                0, 1 - (reduction_factor + renewable_factor)
            )

            gap = baseline - target_emission
            gap_pct = (gap / baseline) * 100

            st.markdown("---")
            st.markdown("### 📊 Results")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Current", f"{baseline:.2f} tCO₂e")
            with c2:
                st.metric(
                    "Target",
                    f"{target_emission:.2f} tCO₂e",
                    delta=f"-{gap_pct:.1f}%",
                    delta_color="inverse"
                )
            with c3:
                st.metric("Reduction Needed", f"{gap:.2f} tCO₂e")

            if target_emission < 20:
                st.success("🟢 **Excellent** – Low-emission target achieved")
            elif target_emission <= 50:
                st.warning("🟡 **Moderate** – Acceptable but improvable")
            else:
                st.error("🔴 **High Emission** – Aggressive action needed")

    # =================================================
    # MODE 2 — MULTI SCENARIO
    # =================================================
    else:

        st.markdown("### 🔮 Compare Multiple Scenarios")

        num = st.slider("Number of Scenarios", 2, 4, 3)

        scenarios = []
        cols = st.columns(num)

        for i, col in enumerate(cols):
            with col:
                st.markdown(f"#### Scenario {i+1}")
                name = st.text_input(
                    "Name",
                    f"Scenario {i+1}",
                    key=f"name_{i}"
                )
                reduction = st.slider(
                    "Reduction %",
                    0.0, 60.0,
                    10.0 * (i+1),
                    key=f"red_{i}"
                )
                renewable = st.slider(
                    "Renewable Gain %",
                    0.0, 50.0,
                    5.0 * (i+1),
                    key=f"ren_{i}"
                )

                scenarios.append((name, reduction, renewable))

        if st.button("⚖️ Compare Scenarios", type="primary"):

            results = []

            for name, r, ren in scenarios:
                factor = (r / 100) + (ren / 200)
                emission = baseline * max(0, 1 - factor)

                results.append({
                    "Scenario": name,
                    "Emission (tCO₂e)": round(emission, 2),
                    "Reduction (%)": r,
                    "Renewable Gain (%)": ren,
                    "Savings": round(baseline - emission, 2)
                })

            df_res = pd.DataFrame(results)
            st.markdown("---")
            st.dataframe(df_res, use_container_width=True)

            best = df_res.loc[df_res["Emission (tCO₂e)"].idxmin()]
            st.success(
                f"🏆 Best Scenario: **{best['Scenario']}** → "
                f"{best['Emission (tCO₂e)']} tCO₂e"
            )

            # -----------------------------
            # BAR CHART
            # -----------------------------
            st.markdown("### 📈 Visual Comparison")

            fig, ax = plt.subplots()
            ax.bar(
                df_res["Scenario"],
                df_res["Emission (tCO₂e)"]
            )
            ax.set_ylabel("Emission (tCO₂e)")
            ax.set_title("Scenario Emission Comparison")
            st.pyplot(fig)
            
            
            st.caption(
                "⚠️ Scenario analysis is **strategic estimation**, "
                "baseline emission is ML-predicted."
            )


            # -----------------------------
            # PROGRESS BARS (SAFE)
            # -----------------------------
            st.markdown("### 📊 Relative Impact")

            for _, row in df_res.iterrows():
                pct = row["Emission (tCO₂e)"] / baseline
                pct = min(max(pct, 0.0), 1.0)

                st.progress(
                    pct,
                    text=f"{row['Scenario']}: {row['Emission (tCO₂e)']} tCO₂e "
                         f"({row['Savings']} saved)"
                )

            st.caption(
                "⚠️ Scenario analysis is **strategic estimation**, "
                "baseline emission is ML-predicted."
            )
