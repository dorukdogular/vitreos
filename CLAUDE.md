# Vitreos — Claude Code Rules

## Project
Glass property prediction ML model. Predicts Tg, density, refractive index from oxide compositions.

## Rules
- No inline comments
- New conversation for debug
- Update memory-bank after every task
- Be extremely token-efficient - no explanations, no summaries, just do the work silently

## Stack
Python 3.11, pandas, numpy, scikit-learn, xgboost, matminer, streamlit, jupyter

## Structure
/data        → raw + cleaned CSVs
/notebooks   → EDA, training, evaluation
/model       → saved model files
/app         → streamlit interface
/memory-bank → project memory