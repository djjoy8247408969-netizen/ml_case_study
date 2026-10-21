# Adult Census Income Classification

A machine-learning classification project that predicts whether an individual's annual income is **greater than $50,000** or **$50,000 or less** using demographic and employment-related attributes from the Adult Census Income dataset.

## Project Overview

This project compares ten classification algorithms, evaluates them on a held-out test set, and uses cross-validation to support model selection.

Target classes:
- `<=50K` — annual income is $50,000 or less
- `>50K` — annual income is greater than $50,000

## Algorithms Evaluated

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree
5. Support Vector Machine (SVM)
6. Random Forest
7. AdaBoost
8. Gradient Boosting
9. Bagging Classifier
10. MLP Neural Network

## Evaluation Metrics

- **Accuracy:** proportion of correctly classified records.
- **Weighted F1-score:** combines precision and recall for each class, weighted by class support.
- **ROC-AUC:** measures how well a model separates the two classes across classification thresholds.

Because the dataset has more records in the `<=50K` class than in the `>50K` class, accuracy is considered alongside F1-score and ROC-AUC.

## Baseline Results

The following values reflect the baseline comparison previously recorded for this project. Verify them against the final results file if the code or evaluation settings have changed.

| # | Algorithm | Accuracy | Weighted F1-score | ROC-AUC |
|---:|---|---:|---:|---:|
| 1 | Logistic Regression | 85.22% | 0.8475 | 0.9077 |
| 2 | K-Nearest Neighbors | 85.66% | 0.8540 | 0.9060 |
| 3 | Gaussian Naive Bayes | 58.97% | 0.6120 | 0.8534 |
| 4 | Decision Tree | 86.55% | 0.8597 | 0.9059 |
| 5 | Support Vector Machine | 85.75% | 0.8516 | 0.9037 |
| 6 | Random Forest | 87.28% | 0.8676 | 0.9251 |
| 7 | AdaBoost | 86.12% | 0.8553 | 0.9160 |
| 8 | Gradient Boosting | **87.78%** | **0.8745** | **0.9318** |
| 9 | Bagging Classifier | 85.02% | 0.8481 | 0.8966 |
| 10 | MLP Neural Network | 85.50% | 0.8538 | 0.9145 |

In this baseline comparison, **Gradient Boosting** achieved the strongest overall results across the listed metrics, followed by Random Forest. Gaussian Naive Bayes had the lowest accuracy and weighted F1-score.

## Algorithm 9 — Bagging Classifier

Bagging stands for *Bootstrap Aggregating*. It trains multiple base estimators on different bootstrap samples of the training data and combines their predictions by voting. The described configuration uses decision trees as base estimators and 100 estimators. Bagging can reduce the variance of an individual decision tree.

Reported baseline test results:
- Accuracy: **85.02%**
- Weighted F1-score: **0.8481**
- ROC-AUC: **0.8966**

## Algorithm 10 — MLP Neural Network

MLP stands for *Multi-Layer Perceptron*. It is a feed-forward neural network that learns patterns by adjusting weights between connected neurons. The described configuration uses one hidden layer with 64 neurons and the ReLU activation function, with early stopping during training.

Reported baseline test results:
- Accuracy: **85.50%**
- Weighted F1-score: **0.8538**
- ROC-AUC: **0.9145**

## Data Preparation

The dataset contains demographic and employment-related features. Typical preparation includes handling categorical and numerical columns separately, encoding categorical variables, and scaling numerical variables where required by the estimator.

**Use the preprocessing pipeline in the source code as the authoritative implementation.** Fit encoders and scalers on training data only; do not fit them on the test set.

## Validation and Model Selection

Cross-validation is used to estimate performance across multiple data splits and help compare models. A stratified train/test split helps preserve target-class proportions.

The project notes also describe tuning Gradient Boosting and lowering its decision threshold from 0.50 to 0.35 to improve recall for the `>50K` class. A lower threshold can increase recall while also increasing false positives. Select the threshold using training/validation data rather than repeatedly optimizing against the final test set.

## Repository Structure

Adapt this example to match the files actually present in your repository:

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── (dataset files or download instructions)
├── notebooks/
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
├── models/
│   └── (optional saved model files)
└── app/
    └── (optional application or deployment code)
```

Remove placeholder entries that do not apply and add your actual filenames.

## Setup

1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies listed in `requirements.txt`.
4. Place the dataset in the location expected by the project, or follow the dataset download instructions.
5. Open and run the relevant notebook or Python entry point.

Create a virtual environment:

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

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducibility Notes

- Use the same train/test split and random seeds to reproduce the comparison.
- Keep preprocessing inside a pipeline where possible to avoid data leakage.
- Report test-set metrics separately from cross-validation metrics.
- Tuned-model results may differ from baseline results because of hyperparameters or threshold changes.

## Limitations and Responsible Use

- The dataset reflects historical census data and may contain sampling, measurement, and societal biases.
- Income prediction is sensitive and should not be used for consequential decisions about individuals without careful validation, fairness analysis, and appropriate safeguards.
- Performance may change when applied to populations or data distributions that differ from the training data.

## License and Dataset Attribution

Before publishing, add the license for your code and the exact dataset citation or source required by the dataset provider.
