# What Makes a Song Go Viral?

**Course:** LBYEC3B – Intelligent Systems for Engineering (De La Salle University – Manila)  
**Group Members:** Dotollo, Melanie S. | Meneses, Jet Ryan S. | Vigo, Karol Miel F.  
**Advisor:** Dr. Edwin Sybingco  

---

## 1. Quick Start

In this project, we built a leakage-free MATLAB pipeline to predict whether a Spotify track will go viral (`1`) or not (`0`) using 11 audio features and 10 music genres across 10,000 unique tracks.

1. Place `run_viral_song_project.m` and `viral_songs.csv` in the same MATLAB folder.
2. Open `run_viral_song_project.m` and click **Run (`F5`)** (or **Publish -> Publish** to export a PDF report).
3. Once finished (~80 seconds), our script automatically saves all trained models, CSV tables, and figures into `models/`, `results/`, and `figures/`.

---

## 2. Models & Lecture Alignment (6 Models Across 5 Families)

We trained and compared **6 models across 5 algorithmic families**. Every model shares the same `rng(42)` seed and stratified 5-fold `cvpartition` during Bayesian hyperparameter tuning:

| # | Model | Family | MATLAB Function | Tuned Hyperparameters |
| :- | :--- | :--- | :--- | :--- |
| 1 | **Logistic Regression** | Linear (**Baseline**) | `fitclinear` (`'logistic'`) | `Lambda`, `Regularization` |
| 2 | **Naive Bayes** | Probabilistic | `fitcnb` | `DistributionNames`, `Width` |
| 3 | **Decision Tree** | Tree | `fitctree` | `MaxNumSplits`, `MinLeafSize` |
| 4 | **Random Forest** | Ensemble (Bagging) | `fitcensemble` (`'Bag'`) | `NumLearningCycles`, `NumVariablesToSample`, `MinLeafSize` |
| 5 | **Gradient Boosted Trees** | Ensemble (Boosting) | `fitcensemble` (`'LogitBoost'`) | `LearnRate`, `NumLearningCycles`, `MaxNumSplits` |
| 6 | **Neural Network** | Neural Network | `fitcnet` | `LayerSizes`, `Activations`, `Lambda` |

---

## 3. Evaluation Protocol Implemented

1. **Leakage-Free Preprocessing & Stratified 5-Fold CV (`cvpartition`, `crossval`, `kfoldPredict`):** We dropped post-release `streams` to prevent target leakage and fitted all imputations, outlier caps, log transforms, and z-score scalers strictly on the 80% training split ($N = 8,000$). Because only **36.05%** of tracks are viral, we chose **F1-score** (`Mean ± SD`) as our primary metric alongside **Precision**, **Recall**, **ROC-AUC**, **Accuracy**, and **Training Time**.
2. **Held-Out Test Evaluation (`predict`, `loss`, `tic`/`toc`):** We evaluated each tuned model once on the untouched 20% test set ($N = 2,000$) and exported the comparison table to `results/model_comparison.csv`.
3. **Statistical Significance Test (`testcholdout`):** We ran a mid-p McNemar's test comparing our top two models on the paired test predictions to check whether their error-rate difference was statistically significant.
4. **Feature Importance & Error Analysis (`predictorImportance`, `confusionmat`, `confusionchart`):** We ranked our best model's top predictors (`results/best_model_feature_importance.csv`) and broke down false positives and false negatives by genre (`results/error_analysis_by_genre.csv`).

---

## 4. Key Takeaways

1. **Logistic Regression Won on Simplicity & Speed:** Logistic Regression achieved the highest Test F1-score (**0.6204**), statistically tying our runner-up Neural Network (**0.6199**, McNemar $p = 0.50$) while training 6× faster (~1.0 s vs. ~6.0 s) and remaining fully interpretable.
2. **Genre Drives Most of the Prediction:** Genre indicators accounted for over **84%** of our model's top decision weight, while `danceability` (**4.43%**) was the strongest individual audio feature.
3. **Genre-Specific Error Patterns:** Mainstream genres like Pop and Hip-Hop produced most of our false positives, whereas niche or traditional genres like Classical, Jazz, and Rock had zero false positives but missed crossover viral hits.