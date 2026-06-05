import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Vitreos", layout="wide")

@st.cache_resource
def load_models():
    with open("model/tg_regressor.pkl", "rb") as f:
        tg_model = pickle.load(f)
    with open("model/density_regressor.pkl", "rb") as f:
        dens_model = pickle.load(f)
    with open("model/tg_features.json") as f:
        tg_features = json.load(f)
    with open("model/density_features.json") as f:
        dens_features = json.load(f)
    return tg_model, dens_model, tg_features, dens_features

tg_model, dens_model, tg_features, dens_features = load_models()

all_oxides = list(dict.fromkeys(tg_features + dens_features))

st.title("Vitreos — Glass Property Predictor")
st.caption("Predict Tg and density from oxide composition")

st.sidebar.header("Oxide Composition (mol %)")
inputs = {}
for ox in all_oxides:
    inputs[ox] = st.sidebar.slider(ox, 0.0, 100.0, 0.0, step=0.1)

total = sum(inputs.values())
if abs(total - 100.0) > 0.5:
    st.sidebar.warning(f"Total: {total:.1f} mol% — should be 100")
else:
    st.sidebar.success(f"Total: {total:.1f} mol%")

if total == 0:
    st.markdown(
        "<div style='text-align:center; margin-top:120px; color:#888; font-size:1.2rem;'>"
        "Enter a composition to predict properties"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    fracs = {k: v / 100.0 for k, v in inputs.items()}

    tg_input = pd.DataFrame([[fracs.get(f, 0.0) for f in tg_features]], columns=tg_features)
    dens_input = pd.DataFrame([[fracs.get(f, 0.0) for f in dens_features]], columns=dens_features)

    tg_pred = tg_model.predict(tg_input)[0]
    dens_pred = dens_model.predict(dens_input)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Glass Transition Temperature")
        st.metric("Tg (K)", f"{tg_pred:.1f} K")
        st.metric("Tg (°C)", f"{tg_pred - 273.15:.1f} °C")
        st.caption("Model confidence: MAE ±32 K (RandomForest, R²=0.89)")

    with col2:
        st.subheader("Density")
        st.metric("Density", f"{dens_pred:.3f} g/cm³")
        st.caption("Model confidence: MAE ±0.21 g/cm³ (RandomForest, R²=0.897)")

    st.divider()
    st.subheader("Input Composition")
    comp_display = {k: v for k, v in inputs.items() if v > 0}
    st.bar_chart(pd.Series(comp_display))
