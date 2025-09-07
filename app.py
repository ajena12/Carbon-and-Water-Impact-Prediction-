import streamlit as st
import pandas as pd
import pickle

# ─────────────────────────────────────────────────────────────
# Load trained model + encoders
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# ─────────────────────────────────────────────────────────────
# Page configuration
st.set_page_config(page_title="Carbon & Water Footprint Predictor", layout="wide")

# Heading
st.markdown(
    "<h2 style='text-align:center; white-space: nowrap;'>🌱 Carbon & Water Footprint Predictor</h2>",
    unsafe_allow_html=True
)

st.markdown("Fill in the product details below and click **Predict Impact**.")

# ─────────────────────────────────────────────────────────────
# Two-column layout for inputs
col1, col2 = st.columns(2, gap="large")

with col1:
    category = st.selectbox("Category", ["Haircare", "Makeup", "Skincare", "Fragrance"])
    material = st.selectbox("Material", ["Plastic", "Glass", "Aluminium", "Paper"])
    packaging_type = st.selectbox("Packaging Type", ["Bottle", "Box", "Tube", "Jar"])
    transport_mode = st.selectbox("Transport Mode", ["Road", "Air", "Sea"])

with col2:
    distance_km = st.slider("Transport Distance (km)", 0, 5000, 100, step=50)
    weight_kg = st.slider("Product Weight (kg)", 0.0, 10.0, 1.0, step=0.1)
    energy_kwh = st.number_input("Energy Used (kWh)", value=5.0, step=0.1)
    water_process_l = st.number_input("Process Water (Liters)", value=50.0, step=1.0)

# ─────────────────────────────────────────────────────────────
if st.button("🔎 Predict Impact"):
    # Build dataframe in same column order as training
    input_dict = {
        "category": [category],
        "material": [material],
        "packaging_type": [packaging_type],
        "transport_mode": [transport_mode],
        "distance_km": [distance_km],
        "weight_kg": [weight_kg],
        "energy_kwh": [energy_kwh],
        "water_process_l": [water_process_l],
    }
    df_input = pd.DataFrame(input_dict)

    # Encode categorical features using saved encoders
    for col, le in label_encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col])

    # Predict (multi-output)
    prediction = model.predict(df_input)[0]
    carbon_pred = prediction[0]
    water_pred  = prediction[1]

    # Display predictions
    st.success(f"🌍 Estimated Carbon Footprint: **{carbon_pred:.2f} kg CO₂e**")
    st.info(f"💧 Estimated Water Usage: **{water_pred:.2f} liters**")

st.markdown("---")
st.caption("Built for L’Oréal Sustainability Challenge 2025")


