# Vitreos 🔬
> Machine learning model that predicts glass transition temperature, density, and refractive index from oxide composition.

## Live Demo
[🚀 Launch Vitreos](https://vitreos.streamlit.app) ← replace with real URL after deploy

## What It Does
Input any oxide glass composition (mol%) → instantly get:
- 🌡️ Glass Transition Temperature (Tg) in K and °C
- ⚖️ Density in g/cm³
- 💡 Refractive Index
- 🧪 Glass Forming Ability probability

## Features
- Dynamic composition builder — add any of 22 oxides, normalize to 100%
- 5 built-in example compositions (soda-lime, borosilicate, fused silica...)
- Compare two glass compositions side by side
- Radar chart visualization
- Export predictions as CSV
- Model card with full training details

## Model Performance
| Property | Samples | R² | MAE |
|---|---|---|---|
| Tg | 76,377 | 0.89 | 32 K |
| Density | 31,173 | 0.90 | 0.21 g/cm³ |
| Refractive Index | 58,913 | 0.92 | 0.021 |
| GFA | 11,858 | — | 67% acc |

## Data
[SciGlass](https://github.com/epam/SciGlass) — 422,000+ inorganic glass compositions, ODbL license.
22 oxide features retained after cleaning (>95% sparsity threshold).

## Run Locally
```bash
git clone https://github.com/dorukdogular/vitreos
cd vitreos
conda create -n vitreos python=3.11
conda activate vitreos
pip install -r app/requirements.txt
streamlit run app/app.py
```

## Known Limitations
- P₂O₅-rich single-component glasses: Tg overestimated
- Heavy-oxide glasses (TeO₂, Bi₂O₃): density underpredicted
- GFA accuracy lower than element-based models (oxide features lose structural info)

## Built By
**Doruk Doğular** · [GitHub](https://github.com/dorukdogular) · 2026

Data from SciGlass under [ODbL](https://opendatacommons.org/licenses/odbl/) license.
