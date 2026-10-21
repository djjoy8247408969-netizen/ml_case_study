# Machine Learning Capstone — Regression, Classification & Clustering

**Course:** 23CSE301 Machine Learning  
**Program:** B.Tech Computer Science and Engineering, III Year  
**Academic Year:** 2026–2027  
**Review:** Capstone Review 2

An end-to-end machine-learning project that explores three different problem types using three UCI datasets. The project covers data auditing, exploratory data analysis (EDA), preprocessing, feature engineering, model training, model comparison, hyperparameter tuning, diagnostics, and deployment through an interactive Streamlit web application.

## Project at a Glance

| Track | Dataset | Task | Best reported model |
|---|---|---|---|
| Regression | Bike Sharing (`hour.csv`) | Predict hourly bike rental demand | Tuned Random Forest Regressor |
| Classification | Adult Census Income | Predict whether annual income is `>50K` | Tuned Gradient Boosting Classifier |
| Clustering | Online Shoppers Purchasing Intention | Discover browsing-behaviour segments without labels | K-Means (`k=4`) |

The notebooks and the accompanying presentation describe the methodology, model comparisons, key results, and deployed application.

## 1. Datasets and Problem Statements

### Track 1 — Regression: Bike Sharing

- **Dataset:** UCI Bike Sharing Dataset (`hour.csv`)
- **Size:** 17,379 hourly records and 17 columns
- **Target:** `cnt`, the number of bikes rented in an hour
- **Goal:** Predict hourly Capital Bikeshare demand from calendar and weather information for Washington, D.C. (2011–2012).
- **Data preparation:** Remove identifier/redundant fields (`instant`, `dteday`) and `casual`/`registered` because those values sum to the target; remove two duplicate rows; create cyclical hour features (`hr_sin`, `hr_cos`) and an `is_rush_hour` feature.
- **Split:** Fixed 80/20 train/test split.

Dataset source: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

### Track 2 — Classification: Adult Census Income

- **Dataset:** UCI Adult / Census Income (`adult.data` and `adult.test`)
- **Size:** 48,842 people and 15 columns
- **Target:** Whether a person's annual income is `>50K` or `<=50K`
- **Goal:** Predict the income class from demographic and employment-related features.
- **Class balance:** Approximately 76% `<=50K` and 24% `>50K`.
- **Data preparation:** Remove 52 duplicate rows; impute missing categorical values using training-set modes; clip `hours-per-week` to `[8, 80]`; engineer `capital_net`; encode categorical variables and scale numerical features through the model pipeline.
- **Split:** Stratified 80/20 split: 39,032 training records and 9,758 test records.

Dataset source: [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)

### Track 3 — Clustering: Online Shoppers

- **Dataset:** UCI Online Shoppers Purchasing Intention
- **Size:** 12,330 browsing sessions and 18 columns
- **Goal:** Discover meaningful visitor segments from browsing behaviour without using purchase labels to fit the clusters.
- **Features used:** `total_pages`, `avg_time_per_page`, `PageValues`, `BounceRates`, `ExitRates`, and `product_share`.
- **Preparation:** Apply `log1p` to compress heavily skewed behaviour features and standardize them before distance-based clustering.
- **Validation:** The `Revenue` label is used after clustering to understand and validate the resulting segments; it is not used to fit the clusters.

Dataset source: [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

## 2. Methodology

Each track follows the same general workflow:

1. **Audit and explore:** Inspect data types, missing values, duplicates, distributions, correlations, and target characteristics.
2. **Clean and engineer features:** Apply track-specific cleaning and create justified features.
3. **Split and preprocess:** Keep preprocessing inside scikit-learn pipelines where applicable; fit imputers, encoders, and scalers on training data only.
4. **Train and compare:** Evaluate the required algorithms using consistent splits and track-specific metrics.
5. **Tune and diagnose:** Use cross-validation and GridSearchCV/RandomizedSearchCV on training data, then evaluate the selected model on the held-out test set.
6. **Save and serve:** Export selected models, metadata, and results for the Streamlit application.

**Leakage prevention:** The presentation and notebooks specify that model selection, preprocessing fitting, hyperparameter tuning, and classification-threshold selection are performed without using the final test set to make those choices.

## 3. Results

The metrics below are the project’s reported held-out test results. Regression and classification use a fixed 80/20 split; classification uses a stratified split. Clustering is evaluated with internal cluster metrics and post-hoc segment validation.

### Regression results

| Rank | Model | R² | RMSE | MAE |
|---:|---|---:|---:|---:|
| 1 | **Random Forest Regressor (tuned)** | **0.9458** | **40.79** | **24.90** |
| 2 | Random Forest Regressor | 0.9292 | 46.63 | 29.79 |
| 3 | Decision Tree Regressor | 0.9052 | 53.96 | 32.59 |
| 4 | SVR (tuned) | 0.9022 | 54.78 | 34.07 |
| 5 | Polynomial Regression | 0.8873 | 58.83 | 42.01 |
| 6 | Gradient Boosting Regressor | 0.8842 | 59.62 | 42.28 |
| 7 | Support Vector Regressor | 0.8726 | 62.54 | 40.11 |
| 8 | K-Nearest Neighbors Regressor | 0.8534 | 67.09 | 45.77 |
| 9 | Lasso Regression | 0.6499 | 103.67 | 78.45 |
| 10 | Ridge Regression | 0.6499 | 103.68 | 78.58 |
| 11 | Linear Regression | 0.6498 | 103.68 | 78.59 |
| 12 | ElasticNet Regression | 0.6441 | 104.52 | 78.46 |

The tuned Random Forest used GridSearchCV and achieved **R² = 0.9458**, **RMSE = 40.79** bikes, and **MAE = 24.90** bikes on the test set.

### Classification results

| Model | Part | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | ROC-AUC |
|---|:---:|---:|---:|---:|---:|---:|
| **Gradient Boosting** | B | **0.8778** | **0.8739** | **0.8778** | **0.8745** | **0.9318** |
| Random Forest | B | 0.8728 | 0.8681 | 0.8728 | 0.8676 | 0.9251 |
| Decision Tree | A | 0.8655 | 0.8601 | 0.8655 | 0.8597 | 0.9059 |
| AdaBoost | B | 0.8612 | 0.8553 | 0.8612 | 0.8553 | 0.9160 |
| K-Nearest Neighbors | A | 0.8566 | 0.8526 | 0.8566 | 0.8540 | 0.9060 |
| MLP Neural Network | B | 0.8550 | 0.8529 | 0.8550 | 0.8538 | 0.9145 |
| Support Vector Machine (SVC) | A | 0.8575 | 0.8512 | 0.8575 | 0.8516 | 0.9037 |
| Bagging (Decision Tree) | B | 0.8502 | 0.8467 | 0.8502 | 0.8481 | 0.8966 |
| Logistic Regression | A | 0.8522 | 0.8462 | 0.8522 | 0.8475 | 0.9077 |
| Gaussian Naive Bayes | A | 0.5897 | 0.8186 | 0.5897 | 0.6120 | 0.8534 |

**Final classifier:** Gradient Boosting tuned with RandomizedSearchCV (`n_estimators=200`, `learning_rate=0.1`, `max_depth=5`, `min_samples_leaf=20`). The reported final test ROC-AUC is **0.9326**. The decision threshold was lowered from `0.50` to `0.35`, selected using out-of-fold training predictions, to increase recall for the `>50K` class from **0.678** to **0.791**. This improves detection of the higher-income class at the cost of more false positives; it does not mean the model has 35% accuracy.

### Clustering results

The selected segmentation uses K-Means with `k=4`.

| Algorithm | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Cluster sizes | Degenerate? |
|---|---:|---:|---:|---|:---:|
| **K-Means (`k=4`)** | 0.3903 | 0.9052 | 7,917 | 7,461 / 2,537 / 1,469 / 863 | No |
| Agglomerative — Ward | 0.3127 | 1.0749 | 6,788 | 6,442 / 2,688 / 2,516 / 684 | No |
| Agglomerative — Complete | 0.3771 | 0.7584 | 3,636 | 10,413 / 1,161 / 737 / 19 | Yes |
| Agglomerative — Average | 0.4328 | 0.5846 | 3,590 | 11,176 / 684 / 451 / 19 | Yes |
| Agglomerative — Single | 0.4624 | 0.3358 | 86 | 12,310 / 18 / 1 / 1 | Yes |

The linkage methods with very high Silhouette scores form tiny clusters and one dominant cluster, so those results are considered degenerate. K-Means was selected as the most useful non-degenerate solution.

#### K-Means visitor segments

| Segment | Sessions | Conversion rate | Share of all purchases |
|---|---:|---:|---:|
| High-Intent Shoppers | 2,537 | **57.9%** | **76.9%** |
| Account & Info Seekers | 1,469 | 8.2% | 6.3% |
| Window Shoppers | 7,461 | 4.2% | 16.6% |
| Bouncers | 863 | 0.3% | 0.2% |

These conversion rates are used to interpret the clusters after they are formed, not as input labels during clustering.

## 4. Algorithms Compared

### Regression

Linear Regression, Ridge Regression, Lasso Regression, ElasticNet, Polynomial Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, Support Vector Regressor (SVR), K-Nearest Neighbors Regressor, plus tuned Random Forest and tuned SVR variants.

### Classification

1. Logistic Regression
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Decision Tree
5. Support Vector Machine (SVC)
6. Random Forest
7. AdaBoost
8. Gradient Boosting
9. Bagging Classifier
10. MLP Neural Network

### Clustering

- K-Means with `k=4`
- Agglomerative Clustering with Ward, Complete, Average, and Single linkage

## 5. Repository Structure

The ZIP contains the project directory `ml-capstone/`. Its main files are organized as follows:

```text
ml-capstone/
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── hour.csv
│   ├── adult.csv
│   ├── prepare_adult.py
│   ├── online_shoppers_intention.csv
│   └── raw_adult/
│       ├── adult.data
│       ├── adult.test
│       └── adult.names
├── notebooks/
│   ├── regression.ipynb
│   ├── classification.ipynb
│   ├── clustering.ipynb
│   └── *.png
├── algorithms/
│   ├── regression/regression.ipynb
│   ├── classification/classification.ipynb
│   └── clustering/clustering.ipynb
├── models/
│   ├── bike_regressor.joblib
│   ├── income_classifier.joblib
│   ├── shopper_segmenter.joblib
│   ├── regression_results.csv
│   ├── classification_results.csv
│   ├── clustering_results.csv
│   └── *_meta.json and supporting result files
└── app/
    ├── app.py
    ├── utils.py
    └── views/
        ├── home.py
        ├── bike.py
        ├── income.py
        ├── shoppers.py
        ├── leaderboard.py
        └── about.py
```

The `notebooks/` directory contains the main analysis notebooks and visual outputs. The `algorithms/` directory also contains track-specific notebooks. The `models/` directory contains saved models, result tables, and metadata consumed by the app.

## 6. Installation and Setup

The project `requirements.txt` pins package versions and specifies Python 3.13 as the tested environment.

### 1. Extract and enter the project directory

```bash
cd ml-capstone
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

To run Jupyter notebooks, install Jupyter if it is not already available:

```bash
pip install jupyter
```

## 7. Run the Project

### Run the Streamlit web application

From the `ml-capstone/` project root:

```bash
streamlit run app/app.py
```

Then open the local URL shown in the terminal, usually:

`http://localhost:8501`

### Run the notebooks

The main notebooks are in `notebooks/`. Open Jupyter from the project root:

```bash
jupyter notebook
```

Run a notebook from top to bottom. Some model-training and hyperparameter-search sections may take several minutes depending on your computer.

### Rebuild the Adult dataset

The processed `data/adult.csv` file is included. To rebuild it from the raw Adult files, run from the project root:

```bash
python data/prepare_adult.py
```

## 8. Streamlit Application Pages

| Page | Functionality |
|---|---|
| Home | Overview of the three tracks and headline model results |
| Bike Demand Forecaster | Enter date, hour, and weather-related values to predict hourly bike rentals; explore a 24-hour curve and what-if changes |
| Income Classifier | Enter an individual's profile to estimate income-class probability; adjust the decision threshold, explore what-if curves, inspect feature importance, and score a batch CSV |
| Shopper Segmentation | Assign browsing behaviour to a segment, show a recommended action and PCA map position, explore segments, and process batch CSVs |
| Model Leaderboards | Display regression, classification, and clustering result tables and charts from saved model outputs |
| About | Project and methodology information |

Saved `.joblib` models are loaded by the app. Use the package versions specified in `requirements.txt`, especially the scikit-learn version used to train the saved models.

## 9. Deployment

The application is designed for Streamlit Community Cloud.

1. Push the `ml-capstone/` directory to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select the repository and branch.
4. Set the main file path to `app/app.py`.
5. Select a compatible Python version (the project specifies Python 3.13 as tested).
6. Deploy and add the resulting public app URL to this README.

**Live app:** Add the deployed URL here after deployment. No live URL is specified in the supplied project files.

## 10. Important Evaluation Metrics

### Regression

- **R²:** proportion of target variance explained by the model; higher is generally better.
- **RMSE:** root mean squared prediction error, in bikes rented per hour; lower is better.
- **MAE:** mean absolute prediction error, in bikes rented per hour; lower is better.

### Classification

- **Accuracy:** fraction of correct predictions.
- **Precision:** among predictions assigned to a class, how many are correct.
- **Recall:** among actual members of a class, how many are found.
- **Weighted F1-score:** class-wise F1-score averaged according to each class's support.
- **ROC-AUC:** how well the model ranks positive examples above negative examples across thresholds.

Because the income classes are imbalanced, the classification comparison uses more than accuracy alone.

### Clustering

- **Silhouette score:** assesses how separated clusters are; higher is generally better.
- **Davies–Bouldin index:** assesses cluster similarity; lower is generally better.
- **Calinski–Harabasz score:** compares between-cluster dispersion with within-cluster dispersion; higher is generally better.

Cluster metrics should be considered together with cluster sizes. A strong score can still describe a practically useless solution if almost all records fall into one cluster.

## 11. Reproducibility and Implementation Notes

- The project uses `random_state=42` for its fixed train/test splits where documented.
- Classification uses a stratified split to preserve class proportions.
- Preprocessing is intended to be fit on training data only.
- Hyperparameters and the classification threshold are selected using training data/cross-validation rather than the final test set.
- `casual` and `registered` are excluded from bike-demand features because they sum to the target `cnt`.
- The clustering fit excludes `Revenue`; purchase behaviour is examined afterwards to interpret the segments.
- The final tuned classifier's metrics differ slightly from the untuned Gradient Boosting row because the final model and decision threshold are different.

## 12. Limitations and Responsible Use

- These results are tied to the supplied datasets, preprocessing choices, splits, and software versions. Performance may differ on new data.
- Census-income predictions reflect historical data and can encode societal or sampling biases. Do not use them for consequential decisions about individuals without rigorous validation, fairness assessment, and appropriate safeguards.
- Shopper segments are statistical groupings, not definitive statements about an individual visitor.
- Forecasts are estimates and should not be treated as guaranteed future outcomes.


## 13. References

- UCI Machine Learning Repository — [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- UCI Machine Learning Repository — [Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- UCI Machine Learning Repository — [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- Project implementation, result CSVs, metadata JSON files, notebooks, and the `ML_Capstone_Review2.pptx` presentation supplied with the project.

---

*Update this README if you change the datasets, source code, saved model files, evaluation split, results, dependency versions, or deployed application URL.*
