# Predicting London House Prices

Predicting what a property in London actually sells for, using the UK
government's official record of every property transaction.

London property is the biggest purchase most people ever make, and pricing is
famously opaque. Asking prices are set by agents with their own incentives, and
buyers rarely have a good independent sense of whether a place is fairly priced.

The goal here is a model that estimates a fair sale price for a London property
from its location and basic characteristics. The kind of thing that could sit
behind a "is this listing over- or under-priced?" tool, or help a buyer sanity-
check an asking price before making an offer.

## The data

Everything is built on the HM Land Registry Price Paid Data which the official
government record of every property sale in England and Wales. It's released
under the Open Government Licence and is the same dataset Rightmove, Zoopla and
the mortgage industry uses.

I used the 2025 yearly file (the most recent complete year), which contains
roughly 800,000 transactions across England and Wales. Filtering to London left
about 72,500 sales to work with.

To capture the effect of transport access which is a big deal in London, I enriched
each property with a public London postcodes reference dataset, which
provides precise coordinates, the transport zone, and the distance to the
nearest station for every postcode.

Both datasets are public but too large to store in this repo. 

Each row is one completed sale, with: price, date, postcode, property type
(flat / terraced / semi / detached), whether it's a new build, and whether it's
freehold or leasehold.

What it crucially does **not** contain: floor area, number of bedrooms,
condition, or remaining lease length. This turns out to matter a lot and is 
a big and annoying limitation for my model.

## The Notebooks

1. Profiling: Checking the shape, types, missing
values and distributions of the raw data.

2. Cleaning: Filtering to London, keeping only standard open-market sales
(dropping repossessions and bulk portfolio deals, which price differently),
bounding prices to a £50k–£10M range, and fixing data types.

3. EDA: Understanding the patterns. The big one: prices are heavily
right-skewed, so the model predicts `log(price)` rather than raw price. One
finding is that properties right next to a station sell for
slightly less than those 250–500m away, because being on top of a station
means noise and crowds.

4. Feature engineering: Building features with an explicit reason for each
one: postcode district (finer than borough), property-type ordering, market
depth per district, and the geographic enrichment (zone, distance to station).

5. Modelling: Comparing three model families with cross-validation:
a linear regression (Ridge), Random Forest, and XGBoost.

6. Explainability: Using SHAP to understand why the model predicts what
it does, both overall and for individual properties.


## Results

Five-fold cross-validation on the training set.

| Model | Cross-val R² | Notes |

| Ridge (linear baseline) | 0.58 | The floor  what the simplest approach achieves |
| Random Forest | 0.69 | Best performer; chosen |
| XGBoost | 0.67 | Strong, but lost to RF with default settings |

Final model: Random Forest, on the held-out test set.

| Metric | Value |

| R² | 0.61 |
| Mean absolute error | £161,000 |
| RMSE | £363,000 |

A few honest notes on these numbers:

- The linear baseline (0.58) being clearly beaten by the tree models (0.69)
  confirms the extra complexity is earning its keep — that non-linear
  distance-to-station effect is exactly the kind of thing trees catch and
  linear models can't.
- Random Forest beat XGBoost, which is slightly unusual — XGBoost usually wins
  on tabular data, but it's more sensitive to hyperparameter tuning, and I
  deliberately compared model *families* with sensible defaults rather than
  tuning one to death. Tuning XGBoost is a clear next step.
- There's a gap between cross-val R² (0.69) and test R² (0.61). I tested
  whether this was overfitting by regularizing the model — it wasn't:
  regularizing dropped both numbers equally without closing the gap, which
  told me the gap is structural variance from the heavy-tailed price
  distribution, not memorization. So I kept the stronger original model.

---

## What drives the predictions

SHAP analysis on a 1,000-property sample, in plain terms:

The single biggest driver is **leasehold vs freehold status**, which I didn't
expect. It makes sense though: in London, tenure is a strong proxy for property
type (leasehold ≈ flat, freehold ≈ house) *and* it captures the genuine value
of owning the land outright.

After that it's **location, location, location** — postcode district, transport
zone, and latitude/longitude. Five of the top eight features are geographic.
Nothing surprising there, but it's nice to see the model quantify what every
estate agent already knows.

One honest finding: a feature I engineered (`is_central`, a simple central-or-
not flag) turned out to be near the bottom of the importance ranking. The zone
and postcode-district features already captured centrality at finer resolution,
making the flag redundant. I left it in because it helps the linear baseline's
interpretability, but a leaner model could drop it. I'd rather report this
honestly than pretend every feature I built was useful.

---

## Limitations

This is the part I'd most want you to read, because knowing what a model
*can't* do matters as much as what it can.

**The big one — no property size.** HM Land Registry records the transaction,
not the property. There's no floor area, no bedroom count, no condition, no
remaining lease length. So a 35m² studio and a 110m² penthouse in the same
building, same type, same tenure, get the *same prediction* — the model
physically cannot tell them apart.

This is the main reason R² caps around 0.61. Studies that use this same dataset
*with* floor area report R² of 0.85–0.90. The ~25-point gap isn't a modelling
failure or a tuning problem — it's an information ceiling. No algorithm can
predict what it can't observe. The fix is better *data*, not a better
algorithm.

**Other honest caveats:**

- Coordinates come from postcode centroids (~15 addresses each), so
  distance-to-station is accurate to roughly ±150m, not to the doorstep.
- It's trained on 2025 only — it captures the 2025 price structure but has no
  concept of market trend and shouldn't be used for multi-year forecasting.
- It's scoped to standard residential sales between £50k and £10M.
  Repossessions, auctions, and ultra-prime property are explicitly out of
  scope; predicting on them would be extrapolation.
- It predicts realised *sale price*, which isn't the same as intrinsic market
  value — a property can sell above or below fair value for all sorts of
  reasons the model can't see.

---

## Future improvements

In rough order of how much they'd help:

1. **Add property floor area via EPC data.** Energy Performance Certificates
   are public and include floor area in m². This is the single highest-impact
   improvement — it's the main thing standing between 0.61 and 0.85.
2. **Hyperparameter tuning**, particularly for XGBoost, which I expect would
   then overtake Random Forest.
3. **An explicit property-type × district interaction feature.**
4. **Multi-year data** to let the model learn market trends.
5. A small web app (Streamlit) so the model can actually be used interactively.

---

## Reproducing this project

```bash
# 1. Clone and set up the environment
git clone <your-repo-url>
cd london-house-price-prediction
pip install -r requirements.txt

# 2. Get the data (it's public but too large to commit)
python src/get_data.py     # prints download instructions

# 3. Run the notebooks in order
#    01 → 02 → 03 → 04 → 05 → 06
#    Each one regenerates the artifacts the next one needs.
```

The two raw files are:

- **HM Land Registry Price Paid Data (2025)** — from gov.uk, save as
  `data/raw/pp-2025.csv`
- **London postcodes reference** — from doogal.co.uk, save as
  `data/raw/london_postcodes.csv`

`src/get_data.py` has the exact links and steps.

---

## Project structure

```
london-house-price-prediction/
├── data/
│   ├── raw/           # downloaded data (not in git — public, large)
│   └── processed/     # regenerated by the notebooks
├── notebooks/
│   ├── 01_data_profiling.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modelling.ipynb
│   └── 06_shap_explain.ipynb
├── src/
│   └── get_data.py    # data download instructions
├── models/            # trained model (not in git — regenerable)
├── reports/figures/   # plots used in this README
├── requirements.txt
├── .gitignore
└── README.md
```

---

## A note on the tools

Built with Python, pandas, scikit-learn, XGBoost and SHAP. The data is official
UK government open data (HM Land Registry, Open Government Licence v3.0) plus a
public London postcodes reference. Nothing here uses paid APIs or private data —
the whole thing is reproducible by anyone from public sources.
