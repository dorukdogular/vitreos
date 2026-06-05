import json, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Vitreos — Glass Property Predictor",
    page_icon="🔬",
    layout="wide",
)

@st.cache_resource
def load_models():
    with open("model/tg_regressor.pkl", "rb") as f:
        tg_model = pickle.load(f)
    with open("model/density_regressor.pkl", "rb") as f:
        dens_model = pickle.load(f)
    with open("model/ri_regressor.pkl", "rb") as f:
        ri_model = pickle.load(f)
    with open("model/gfa_classifier_v2.pkl", "rb") as f:
        gfa_model = pickle.load(f)
    with open("model/tg_features.json") as f:
        tg_features = json.load(f)
    with open("model/density_features.json") as f:
        dens_features = json.load(f)
    with open("model/ri_features.json") as f:
        ri_features = json.load(f)
    return tg_model, dens_model, ri_model, gfa_model, tg_features, dens_features, ri_features

tg_model, dens_model, ri_model, gfa_model, tg_features, dens_features, ri_features = load_models()

ALL_OXIDES = ["SiO2","P2O5","ZrO2","Na2O","Al2O3","Fe2O3","CaO","MgO","K2O","B2O3","BaO","ZnO","Li2O","SrO","La2O3","TiO2","Nb2O5","PbO","Sb2O3","Bi2O3","TeO2","Se"]

st.title("Vitreos — Glass Property Predictor")
st.caption("Predict Tg and density from oxide composition")
st.markdown(
    "<p style='color:#999; font-size:0.85rem; margin-top:-8px;'>"
    "Trained on 76,000+ glass compositions from the SciGlass database"
    "</p>",
    unsafe_allow_html=True,
)

if "composition" not in st.session_state:
    st.session_state.composition = {}

st.sidebar.header("Composition Builder")

remaining = [ox for ox in ALL_OXIDES if ox not in st.session_state.composition]

add_col, btn_col = st.sidebar.columns([3, 1])
with add_col:
    selected_oxide = st.selectbox(
        "Oxide",
        options=remaining if remaining else ["—"],
        label_visibility="collapsed",
    )
with btn_col:
    st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
    if st.button("Add", use_container_width=True):
        if selected_oxide and selected_oxide != "—" and selected_oxide not in st.session_state.composition:
            st.session_state.composition[selected_oxide] = 0.0
            st.rerun()

oxides_to_remove = []
for oxide in list(st.session_state.composition.keys()):
    row_left, row_right = st.sidebar.columns([4, 1])
    with row_left:
        st.number_input(
            oxide,
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.composition[oxide],
            step=0.1,
            key=f"val_{oxide}",
        )
        st.session_state.composition[oxide] = st.session_state[f"val_{oxide}"]
    with row_right:
        st.markdown("<div style='margin-top:26px;'></div>", unsafe_allow_html=True)
        if st.button("✕", key=f"rm_{oxide}"):
            oxides_to_remove.append(oxide)

if oxides_to_remove:
    for ox in oxides_to_remove:
        del st.session_state.composition[ox]
    st.rerun()

total = sum(st.session_state.composition.values())

if total == 0:
    st.sidebar.markdown("<span style='color:#aaa;'>0.0 mol%</span>", unsafe_allow_html=True)
elif abs(total - 100) <= 0.5:
    st.sidebar.success(f"✓ {total:.1f} mol%")
else:
    st.sidebar.warning(f"⚠ {total:.1f} mol% — should be 100")

norm_col, clear_col = st.sidebar.columns(2)
with norm_col:
    if st.button("Normalize to 100%", use_container_width=True):
        if total > 0:
            for ox in st.session_state.composition:
                st.session_state.composition[ox] = (st.session_state.composition[ox] / total) * 100
            st.rerun()
with clear_col:
    if st.button("Clear all", use_container_width=True):
        st.session_state.composition = {}
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='color:#aaa; font-size:0.8rem; line-height:1.6;'>"
    "<strong>Vitreos v2.0</strong><br>"
    "Doruk Doğular · 2026<br>"
    "<a href='https://github.com/dorukdogular/vitreos' style='color:#aaa;'>github.com/dorukdogular/vitreos</a>"
    "</div>",
    unsafe_allow_html=True,
)

inputs = st.session_state.composition
total = sum(inputs.values())

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
    ri_input = pd.DataFrame([[fracs.get(f, 0.0) for f in ri_features]], columns=ri_features)
    ri_pred = ri_model.predict(ri_input)[0]
    gfa_input = pd.DataFrame([[fracs.get(f, 0.0) for f in tg_features]], columns=tg_features)
    gfa_prob = float(gfa_model.predict_proba(gfa_input)[0][1])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Tg")
        st.metric("Tg (K)", f"{tg_pred:.1f} K")
        st.metric("Tg (°C)", f"{tg_pred - 273.15:.1f} °C")
        st.caption("MAE ±32 K (RF, R²=0.89)")

    with col2:
        st.subheader("Density")
        st.metric("Density", f"{dens_pred:.3f} g/cm³")
        st.caption("MAE ±0.21 g/cm³ (RF, R²=0.897)")

    with col3:
        st.subheader("Refractive Index")
        st.metric("RI", f"{ri_pred:.4f}")
        st.caption("MAE ±0.021 (RF, R²=0.922)")

    with col4:
        st.subheader("GFA")
        gfa_color = "green" if gfa_prob > 0.7 else ("orange" if gfa_prob >= 0.4 else "red")
        st.markdown(
            f"<div style='font-size:2rem; color:{gfa_color}; font-weight:bold; margin-bottom:4px;'>{gfa_prob*100:.0f}%</div>",
            unsafe_allow_html=True,
        )
        st.caption("RF classifier (acc=0.67, oxide features)")

    tg_norm = float(np.clip((tg_pred - 300) / (1500 - 300), 0, 1))
    dens_norm = float(np.clip((dens_pred - 1.5) / (8.0 - 1.5), 0, 1))
    ri_norm = float(np.clip((ri_pred - 1.3) / (2.5 - 1.3), 0, 1))
    gfa_norm = gfa_prob

    categories = ["Tg", "Density", "GFA", "Refractive Index"]
    values = [tg_norm, dens_norm, gfa_norm, ri_norm]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        line_color="#4C9BE8",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Input Composition")
    st.bar_chart(pd.Series({k: v for k, v in inputs.items() if v > 0}))

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.8rem; padding:8px 0;'>"
    "Built by <a href='https://github.com/dorukdogular' style='color:#aaa;'>Doruk Doğular</a> · "
    "<a href='https://github.com/dorukdogular/vitreos' style='color:#aaa;'>⭐ vitreos on GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
