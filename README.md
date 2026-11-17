# What Makes a Song Go Viral? — Predicting Spotify Track Virality

**Course:** LBYEC3B – Intelligent Systems for Engineering (De La Salle University – Manila)  
**Group Members:** Dotollo, Melanie S. | Meneses, Jet Ryan S. | Vigo, Karol Miel F.  
**Advisor:** Dr. Edwin Sybingco  

---

## 1. Environment & Reproducibility Requirements

* **MATLAB Release:** MATLAB R2021a or newer (required for `fitcnet`, `boxchart`, and `exportgraphics`; tested on R2024a / R2024b)
* **Required Toolboxes:** Statistics and Machine Learning Toolbox
* **Fixed Random Seed:** `SEED = 42` initialized via `rng(42, 'twister')` in `main.m` and passed across data splitting, 5-fold cross-validation partitioning, Bayesian hyperparameter optimization, and permutation importance

---

## 2. Repository Architecture & Module Ownership

```text
LBYEC3B_ViralSong_Project/
├── main.m                          # End-to-end pipeline runner (Click Run / F5 here)
├── trainModels.m                   # Bayesian optimization (5-fold CV), hyperparameter log & model saving
├── README.md                       # Environment, workflow & model documentation
│
├── prepData.m                      # Leakage-free preprocessing & stratified splitting (untouched)
├── testPrepData.m                  # 12-point automated preprocessing & leakage verification (untouched)
├── edaFigures.m                    # Exploratory data analysis figures & data dictionary (untouched)
├── evalModels.m                    # CV/test evaluation, McNemar's test, importance & error PDFs (untouched)
│
├── viral_songs.csv                 # Input dataset (10,000 tracks)
├── LBYEC3B_ViralSong_Project.prj   # MATLAB Project file
├── models/                         # Saved trained models & hyperparameter log
├── figures/                        # Exploratory data analysis figures & summary CSVs
└── results/                        # Evaluation tables, vector PDFs & evalResults.mat
```

---

## 3. Quick Start (`main.m`)

1. Open `LBYEC3B_ViralSong_Project.prj` (or set the repository root folder as your MATLAB working directory).
2. Open **`main.m`** and click **Run (`F5`)**.
3. `main.m` executes all five stages end-to-end from `SEED = 42`:
   * **Step 1 (`testPrepData.m`):** Runs 12 automated verification checks confirming zero target or statistical leakage.
   * **Step 2 (`prepData.m`):** Performs an 80/20 stratified train/test split ($N = 8,000$ train / $2,000$ test) and fits training-only median/mode imputation, 1st/99th percentile capping, `log1p` transforms, z-score standardization, and one-hot genre encoding (21 predictors total).
   * **Step 3 (`edaFigures.m`):** Exports exploratory figures (`figures/*.pdf`, `figures/*.png`) and summary tables (`figures/data_dictionary.csv`, `figures/eda_feature_summary.csv`, `figures/eda_genre_summary.csv`).
   * **Step 4 (`trainModels.m`):** Tunes the baseline and 5 models across 5 algorithmic families using Bayesian optimization with 5-fold CV on the training set, logs optimal hyperparameters to `models/tuned_hyperparameters.csv`, saves trained models to `models/trained_models.mat`, and returns `cfg`.
   * **Step 5 (`evalModels.m`):** Runs 5-fold CV (`Mean ± SD`), single held-out test-set evaluation, McNemar's test (best vs. runner-up), permutation feature importance, and error analysis, exporting all tables (`results/tables/*.csv`) and vector PDFs (`results/figures/*.pdf`).

---

## 4. Models & Tuned Hyperparameter Search Spaces

All models share the same `SEED = 42` (`'twister'`) generator and stratified 5-fold `cvpartition` during Bayesian optimization (`expected-improvement-plus`):

| # | Model | Family | MATLAB Function | Tuned Hyperparameters & Search Ranges |
| :- | :--- | :--- | :--- | :--- |
| 1 | **Logistic Regression** | Linear (**Baseline**) | `fitclinear` (`'logistic'`) | `Lambda` $[1.25\times10^{-9},\, 12.5]$, `Regularization` `{'ridge', 'lasso'}` |
| 2 | **Naive Bayes** | Probabilistic | `fitcnb` | `DistributionNames` `{'normal', 'kernel'}`, `Width` $[0.30,\, 1.50]$ |
| 3 | **Decision Tree** | Tree | `fitctree` | `MaxNumSplits` $[1,\, 7999]$, `MinLeafSize` $[1,\, 4000]$ |
| 4 | **Random Forest** | Ensemble (Bagging) | `fitcensemble` (`'Bag'`) | `NumLearningCycles` $[20,\, 60]$, `MinLeafSize` $[10,\, 40]$, `NumVariablesToSample` $[3,\, 10]$ |
| 5 | **Gradient Boosted Trees** | Ensemble (Boosting) | `fitcensemble` (`'LogitBoost'`) | `NumLearningCycles` $[20,\, 60]$, `LearnRate` $[0.03,\, 0.25]$, `MaxNumSplits` $[2,\, 12]$ |
| 6 | **Neural Network** | Neural Network | `fitcnet` | `Layer_1_Size` $[8,\, 64]$, `Activations` `{'relu', 'tanh', 'sigmoid'}`, `Lambda` $[10^{-5},\, 10^{-1}]$ |

---

## 5. Dataset & Feature Summary (`viral_songs.csv`)

The dataset comprises **10,000 unique Spotify tracks** with a **36.05% viral rate**:

| Variable | Type | Role | Preprocessing / Notes |
| :--- | :--- | :--- | :--- |
| `track_id`, `track_name`, `artist` | string | Dropped | Identifier columns removed prior to modeling |
| `streams` | integer | Excluded | Post-release play count excluded to prevent target leakage |
| `release_year` | integer | Predictor (1) | Z-score standardized (not percentile-capped) |
| `duration_sec`, `danceability`, `energy`, `valence`, `tempo_bpm`, `liveness`, `loudness_db` | numeric | Predictors (2–8) | Capped at training 1st/99th percentiles + z-score standardized |
| `acousticness`, `instrumentalness`, `speechiness` | float | Predictors (9–11) | Capped at training 1st/99th percentiles + `log1p` + z-score standardized |
| `genre` | categorical | Predictors (12–21) | 10 genres one-hot encoded (`Classical`, `Country`, `Electronic`, `Hip-Hop`, `Indie`, `Jazz`, `Latin`, `Pop`, `R&B`, `Rock`) |
| `viral` | binary (`0`/`1`) | **Target** | `1` = Viral hit (36.05%), `0` = Non-viral track (63.95%) |