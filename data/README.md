# Data

## Source

Glass property data is sourced from the [SciGlass database](https://github.com/epam/SciGlass) (EPAM Systems), made available under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/). Accessed via the [glasspy](https://github.com/drcassar/glasspy) Python library.

## Files tracked in git

| File | Description |
|---|---|
| `clean.csv` | Deduplicated glass-forming ability dataset (11858 rows) |
| `tg_dataset.csv` | 76377 glasses with Tg (K) and 22 oxide mol% features |
| `tg_density_dataset.csv` | 31173 glasses with both Tg (K) and density (g/cm³) |

## Regenerating all data

Run notebooks in order from the project root using the `vitreos` conda environment:

```bash
conda run -n vitreos python notebooks/01_data_collection.py
conda run -n vitreos python data/clean.py
conda run -n vitreos python notebooks/02_feature_engineering.py
conda run -n vitreos python notebooks/03_train_classifier.py
conda run -n vitreos python notebooks/04_clean_properties.py
conda run -n vitreos python notebooks/05_train_tg_regressor.py
conda run -n vitreos python notebooks/06_train_density_regressor.py
conda run -n vitreos python notebooks/07_evaluation.py
```

`data/sciglass_raw/` and model `.pkl` files are excluded from git (regenerable).
