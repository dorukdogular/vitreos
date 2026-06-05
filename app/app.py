import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Vitreos — Glass Property Predictor",
    page_icon="🔬",
    layout="wide",
)

@st.cache_resource
def load_models():
    tg_model   = joblib.load("model/tg_regressor.pkl")
    dens_model = joblib.load("model/density_regressor.pkl")
    ri_model   = joblib.load("model/ri_regressor.pkl")
    gfa_model  = joblib.load("model/gfa_classifier_v2.pkl")
    with open("model/tg_features.json") as f:
        tg_features = json.load(f)
    with open("model/density_features.json") as f:
        dens_features = json.load(f)
    with open("model/ri_features.json") as f:
        ri_features = json.load(f)
    return tg_model, dens_model, ri_model, gfa_model, tg_features, dens_features, ri_features

tg_model, dens_model, ri_model, gfa_model, tg_features, dens_features, ri_features = load_models()

@st.cache_resource
def get_supabase():
    from supabase import create_client
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

ALL_OXIDES = ["SiO2","P2O5","ZrO2","Na2O","Al2O3","Fe2O3","CaO","MgO","K2O","B2O3",
              "BaO","ZnO","Li2O","SrO","La2O3","TiO2","Nb2O5","PbO","Sb2O3","Bi2O3","TeO2","Se"]

EXAMPLES = {
    "Soda-Lime Glass":    {"SiO2": 72, "Na2O": 14, "CaO": 9, "MgO": 4, "Al2O3": 1},
    "Borosilicate (Pyrex)": {"SiO2": 81, "B2O3": 13, "Na2O": 4, "Al2O3": 2},
    "Fused Silica":       {"SiO2": 100},
    "Lead Crystal":       {"SiO2": 54, "PbO": 29, "K2O": 12, "Na2O": 5},
    "Aluminosilicate":    {"SiO2": 57, "Al2O3": 16, "CaO": 8, "MgO": 7, "Na2O": 7, "B2O3": 5},
}


def predict_all(fracs):
    tg_in   = pd.DataFrame([[fracs.get(f, 0.0) for f in tg_features]],   columns=tg_features)
    dens_in = pd.DataFrame([[fracs.get(f, 0.0) for f in dens_features]], columns=dens_features)
    ri_in   = pd.DataFrame([[fracs.get(f, 0.0) for f in ri_features]],   columns=ri_features)
    gfa_in  = pd.DataFrame([[fracs.get(f, 0.0) for f in tg_features]],   columns=tg_features)
    return (
        float(tg_model.predict(tg_in)[0]),
        float(dens_model.predict(dens_in)[0]),
        float(ri_model.predict(ri_in)[0]),
        float(gfa_model.predict_proba(gfa_in)[0][1]),
    )


def normalize_values(vals_raw):
    tg, dens, ri, gfa = vals_raw
    return [
        float(np.clip((tg   - 300)  / (1500 - 300),  0, 1)),
        float(np.clip((dens - 1.5)  / (8.0  - 1.5),  0, 1)),
        gfa,
        float(np.clip((ri   - 1.3)  / (2.5  - 1.3),  0, 1)),
    ]


def render_builder(key_prefix, label):
    state_key = f"comp_{key_prefix}"
    ex_key    = f"ex_{key_prefix}"

    def _load_example():
        ex = st.session_state[ex_key]
        if ex != "— Select —":
            st.session_state[state_key] = dict(EXAMPLES[ex])
            st.session_state[ex_key]    = "— Select —"

    st.subheader(label)
    st.selectbox(
        "Load Example",
        options=["— Select —"] + list(EXAMPLES.keys()),
        key=ex_key,
        on_change=_load_example,
    )

    comp = st.session_state[state_key]
    remaining_local = [ox for ox in ALL_OXIDES if ox not in comp]
    add_c, btn_c = st.columns([3, 1])
    with add_c:
        sel = st.selectbox(
            "Oxide",
            options=remaining_local if remaining_local else ["—"],
            label_visibility="collapsed",
            key=f"sel_{key_prefix}",
        )
    with btn_c:
        st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
        if st.button("Add", key=f"add_{key_prefix}", use_container_width=True):
            if sel and sel != "—" and sel not in comp:
                comp[sel] = 0.0
                st.rerun()

    to_remove = []
    for oxide in list(comp.keys()):
        r_left, r_right = st.columns([4, 1])
        with r_left:
            st.number_input(
                oxide, min_value=0.0, max_value=100.0,
                value=comp[oxide], step=0.1,
                key=f"val_{key_prefix}_{oxide}",
            )
            comp[oxide] = st.session_state[f"val_{key_prefix}_{oxide}"]
        with r_right:
            st.markdown("<div style='margin-top:26px;'></div>", unsafe_allow_html=True)
            if st.button("✕", key=f"rm_{key_prefix}_{oxide}"):
                to_remove.append(oxide)

    if to_remove:
        for ox in to_remove:
            del comp[ox]
        st.rerun()

    total_c = sum(comp.values())
    n_col, c_col = st.columns(2)
    with n_col:
        if st.button("Normalize", key=f"norm_{key_prefix}", use_container_width=True):
            if total_c > 0:
                for ox in comp:
                    normalized = (comp[ox] / total_c) * 100
                    comp[ox] = normalized
                    st.session_state[f"val_{key_prefix}_{ox}"] = normalized
                st.rerun()
    with c_col:
        if st.button("Clear", key=f"clear_{key_prefix}", use_container_width=True):
            st.session_state[state_key] = {}
            st.rerun()

    if total_c == 0:
        st.markdown("<span style='color:#aaa;'>0.0 mol%</span>", unsafe_allow_html=True)
    elif abs(total_c - 100) <= 0.5:
        st.success(f"✓ {total_c:.1f} mol%")
    else:
        st.warning(f"⚠ {total_c:.1f} mol% — should be 100")

    return comp


# ── PAGE HEADER ──────────────────────────────────────────────────────────────
st.title("Vitreos — Glass Property Predictor")
st.caption("Predict Tg, density, refractive index, and GFA from oxide composition")
st.markdown(
    "<p style='color:#999; font-size:0.85rem; margin-top:-8px;'>"
    "Trained on 76,000+ glass compositions from the SciGlass database"
    "</p>",
    unsafe_allow_html=True,
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for _k in ("composition", "comp_a", "comp_b"):
    if _k not in st.session_state:
        st.session_state[_k] = {}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
try:
    sb = get_supabase()
    st.sidebar.success("DB connected")
except Exception as e:
    st.sidebar.error(f"DB connection failed: {e}")

st.sidebar.header("Composition Builder")
st.sidebar.caption("Used in the Single Glass tab")

def _load_single_example():
    ex = st.session_state["ex_single"]
    if ex != "— Select —":
        st.session_state["composition"] = dict(EXAMPLES[ex])
        st.session_state["ex_single"]   = "— Select —"

st.sidebar.selectbox(
    "Load Example",
    options=["— Select —"] + list(EXAMPLES.keys()),
    key="ex_single",
    on_change=_load_single_example,
)

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
            oxide, min_value=0.0, max_value=100.0,
            value=st.session_state.composition[oxide], step=0.1,
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

total_sidebar = sum(st.session_state.composition.values())
if total_sidebar == 0:
    st.sidebar.markdown("<span style='color:#aaa;'>0.0 mol%</span>", unsafe_allow_html=True)
elif abs(total_sidebar - 100) <= 0.5:
    st.sidebar.success(f"✓ {total_sidebar:.1f} mol%")
else:
    st.sidebar.warning(f"⚠ {total_sidebar:.1f} mol% — should be 100")

norm_col, clear_col = st.sidebar.columns(2)
with norm_col:
    if st.button("Normalize to 100%", use_container_width=True):
        if total_sidebar > 0:
            for ox in st.session_state.composition:
                normalized = (st.session_state.composition[ox] / total_sidebar) * 100
                st.session_state.composition[ox] = normalized
                st.session_state[f"val_{ox}"] = normalized
            st.rerun()
with clear_col:
    if st.button("Clear all", use_container_width=True):
        st.session_state.composition = {}
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='color:#aaa; font-size:0.8rem; line-height:1.6;'>"
    "<strong>Vitreos v2.1</strong><br>"
    "Doruk Doğular · 2026<br>"
    "<a href='https://github.com/dorukdogular/vitreos' style='color:#aaa;'>github.com/dorukdogular/vitreos</a>"
    "</div>",
    unsafe_allow_html=True,
)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Single Glass", "Compare Two Glasses", "Model Card"])

# ─────────────────────────── TAB 1 ───────────────────────────────────────────
with tab1:
    inputs  = st.session_state.composition
    total_t1 = sum(inputs.values())

    if total_t1 == 0:
        st.markdown(
            "<div style='text-align:center; margin-top:120px; color:#888; font-size:1.2rem;'>"
            "Enter a composition to predict properties"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        fracs = {k: v / 100.0 for k, v in inputs.items()}
        tg_pred, dens_pred, ri_pred, gfa_prob = predict_all(fracs)

        print("INPUT TYPE:", type(inputs))
        print("INPUT VALUE:", inputs)
        if st.session_state.get("last_logged") != dict(inputs):
            try:
                sb = get_supabase()
                sb.table("predictions").insert({
                    "composition":      dict(inputs),
                    "tg_k":             round(tg_pred, 2),
                    "tg_c":             round(tg_pred - 273.15, 2),
                    "density":          round(dens_pred, 4),
                    "refractive_index": round(ri_pred, 4),
                    "gfa_pct":          round(gfa_prob * 100, 1),
                }).execute()
                st.session_state["last_logged"] = dict(inputs)
                print("Supabase insert OK:", dict(inputs))
            except Exception as e:
                import traceback
                print("SUPABASE FULL ERROR:")
                print(traceback.format_exc())

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.subheader("Tg")
            st.metric("Tg (K)",  f"{tg_pred:.1f} K")
            st.metric("Tg (°C)", f"{tg_pred - 273.15:.1f} °C")
            st.caption("MAE ±32 K (RF, R²=0.89)")
        with col2:
            st.subheader("Density")
            st.metric("Density", f"{dens_pred:.3f} g/cm³")
            st.caption("MAE ±0.21 g/cm³ (RF, R²=0.90)")
        with col3:
            st.subheader("Refractive Index")
            st.metric("RI", f"{ri_pred:.4f}")
            st.caption("MAE ±0.021 (RF, R²=0.922)")
        with col4:
            st.subheader("GFA")
            gfa_color = "green" if gfa_prob > 0.7 else ("orange" if gfa_prob >= 0.4 else "red")
            st.markdown(
                f"<div style='font-size:2rem; color:{gfa_color}; font-weight:bold; margin-bottom:4px;'>"
                f"{gfa_prob*100:.0f}%</div>",
                unsafe_allow_html=True,
            )
            st.caption("RF classifier (acc=0.67, oxide features)")

        norms = normalize_values((tg_pred, dens_pred, ri_pred, gfa_prob))
        categories  = ["Tg", "Density", "GFA", "Refractive Index"]
        vals_closed = norms + [norms[0]]
        cats_closed = categories + [categories[0]]

        fig = go.Figure(go.Scatterpolar(
            r=vals_closed, theta=cats_closed, fill="toself",
            line_color="#2563eb",
            fillcolor="rgba(37,99,235,0.15)",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor="#cbd5e1", tickfont=dict(color="#0f172a"),
                ),
                angularaxis=dict(
                    gridcolor="#cbd5e1", tickfont=dict(color="#0f172a"),
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0f172a", size=13),
            showlegend=False, height=350, margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Input Composition")
        bar_data = pd.Series({k: v for k, v in inputs.items() if v > 0}).reset_index()
        bar_data.columns = ["Oxide", "mol%"]
        fig_bar = px.bar(
            bar_data, x="Oxide", y="mol%",
            color_discrete_sequence=["#2563eb"],
            template="plotly_white",
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0f172a"),
            margin=dict(l=0, r=0, t=20, b=0),
        )
        fig_bar.update_xaxes(gridcolor="#cbd5e1", tickfont=dict(color="#0f172a"))
        fig_bar.update_yaxes(gridcolor="#cbd5e1", tickfont=dict(color="#0f172a"))
        st.plotly_chart(fig_bar, use_container_width=True)

        row = {ox: inputs.get(ox, 0.0) for ox in ALL_OXIDES}
        row.update({
            "Tg_K":          round(tg_pred, 2),
            "Tg_C":          round(tg_pred - 273.15, 2),
            "Density_gcm3":  round(dens_pred, 4),
            "RI":            round(ri_pred, 4),
            "GFA_pct":       round(gfa_prob * 100, 1),
        })
        st.download_button(
            label="Download Results",
            data=pd.DataFrame([row]).to_csv(index=False),
            file_name="vitreos_prediction.csv",
            mime="text/csv",
        )


# ─────────────────────────── TAB 2 ───────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        comp_a = render_builder("a", "Glass A")
    with col_b:
        comp_b = render_builder("b", "Glass B")

    total_a = sum(comp_a.values())
    total_b = sum(comp_b.values())

    if total_a == 0 or total_b == 0:
        st.info("Enter compositions for both Glass A and Glass B to compare.")
    else:
        fracs_a = {k: v / 100.0 for k, v in comp_a.items()}
        fracs_b = {k: v / 100.0 for k, v in comp_b.items()}

        tg_a, dens_a, ri_a, gfa_a = predict_all(fracs_a)
        tg_b, dens_b, ri_b, gfa_b = predict_all(fracs_b)

        compare_df = pd.DataFrame({
            "Property": ["Tg (K)", "Tg (°C)", "Density (g/cm³)", "Refractive Index", "GFA (%)"],
            "Glass A":  [f"{tg_a:.1f}", f"{tg_a-273.15:.1f}", f"{dens_a:.3f}", f"{ri_a:.4f}", f"{gfa_a*100:.1f}"],
            "Glass B":  [f"{tg_b:.1f}", f"{tg_b-273.15:.1f}", f"{dens_b:.3f}", f"{ri_b:.4f}", f"{gfa_b*100:.1f}"],
            "A − B":    [
                f"{tg_a   - tg_b:+.1f}",
                f"{tg_a   - tg_b:+.1f}",
                f"{dens_a - dens_b:+.3f}",
                f"{ri_a   - ri_b:+.4f}",
                f"{(gfa_a - gfa_b)*100:+.1f}",
            ],
        })
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        norms_a = normalize_values((tg_a, dens_a, ri_a, gfa_a))
        norms_b = normalize_values((tg_b, dens_b, ri_b, gfa_b))
        categories  = ["Tg", "Density", "GFA", "Refractive Index"]
        cats_closed = categories + [categories[0]]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=norms_a + [norms_a[0]], theta=cats_closed, fill="toself",
            line_color="#2563eb", fillcolor="rgba(37,99,235,0.15)",
            name="Glass A",
        ))
        fig2.add_trace(go.Scatterpolar(
            r=norms_b + [norms_b[0]], theta=cats_closed, fill="toself",
            line_color="#f97316", fillcolor="rgba(249,115,22,0.15)",
            name="Glass B",
        ))
        fig2.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor="#cbd5e1", tickfont=dict(color="#0f172a"),
                ),
                angularaxis=dict(
                    gridcolor="#cbd5e1", tickfont=dict(color="#0f172a"),
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0f172a", size=13),
            showlegend=True, height=400, margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────── TAB 3 ───────────────────────────────────────────
with tab3:
    st.markdown("""
## Vitreos Model Card

**Training Data:** SciGlass database ([epam/SciGlass](https://github.com/epam/SciGlass), ODbL license), 422,000+ inorganic glasses.

**Models:** RandomForestClassifier / RandomForestRegressor (scikit-learn).

**Feature space:** 22 oxide mol% columns — SiO₂, P₂O₅, ZrO₂, Na₂O, Al₂O₃, Fe₂O₃, CaO, MgO, K₂O, B₂O₃, BaO, ZnO, Li₂O, SrO, La₂O₃, TiO₂, Nb₂O₅, PbO, Sb₂O₃, Bi₂O₃, TeO₂, Se.

### Performance

| Property | Training Samples | R² | MAE |
|---|---|---|---|
| Tg | 76,377 | 0.89 | ±32 K |
| Density | 31,173 | 0.90 | ±0.21 g/cm³ |
| Refractive Index | 58,913 | 0.92 | ±0.021 |
| GFA | 11,858 | — | 67% accuracy |

### Known Limitations

- **P₂O₅-rich glasses:** Tg tends to be overestimated; these glasses have complex network structures underrepresented in training data.
- **Heavy-oxide glasses (TeO₂, Bi₂O₃):** Density is underpredicted; these compositions are sparse in SciGlass.
- **GFA accuracy:** Reduced vs element-based model (0.67 vs 0.90). Converting to oxide mol% loses element-level structural information needed for glass formation prediction.

### About

Built by [Doruk Doğular](https://github.com/dorukdogular) · 2026.
Source: [github.com/dorukdogular/vitreos](https://github.com/dorukdogular/vitreos)
Training data: SciGlass open database under [ODbL license](https://opendatacommons.org/licenses/odbl/1-0/).
""")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.8rem; padding:8px 0;'>"
    "Built by <a href='https://github.com/dorukdogular' style='color:#aaa;'>Doruk Doğular</a> · "
    "<a href='https://github.com/dorukdogular/vitreos' style='color:#aaa;'>⭐ vitreos on GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
