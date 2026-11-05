# Cost-Sensitive Heavy Industrial Air Pressure System (APS) Component Failure Prediction

## Overview
This repository contains the official codebase and technical documentation for the course project of **DS605: Fundamentals of Machine Learning** under the guidance of Dr. Arpit Rana at DA-IICT. 

The project implements an end-to-end machine learning pipeline for predictive maintenance and fault diagnosis of heavy-duty commercial vehicle Air Pressure Systems (APS) using the UCI Scania APS Failure dataset. The primary objective is to build, evaluate, and optimize classification models that minimize operational maintenance costs by balancing expensive missed component breakdowns against unnecessary workshop inspections.

---

## Team Information
* **Team ID / Members:** 
  * `202618052` (Lakshita Pagaria)
  * `202618047` (Ameya Trivedi)
* **Course:** DS605 Fundamentals of Machine Learning (Autumn 2026-27)
* **Institution:** DA-IICT (Dhirubhai Ambani Institute of Information and Communication Technology)

---

## Dataset Description
The project utilizes the **APS Failure at Scania Trucks Dataset** obtained from the UCI Machine Learning Repository.
* **Training Set:** 60,000 operational records $\times$ 171 columns
* **Test Set:** 16,000 operational records $\times$ 171 columns
* **Features:** 170 anonymized continuous sensor counters and multi-bin operational histograms, alongside 1 binary target label (`neg` for normal operating cycles vs. `pos` for APS component failures).
* **Official Industrial Cost Function:** 
  $$\text{Total Maintenance Cost} = (\$10 \times \text{False Positives}) + (\$500 \times \text{False Negatives})$$
  *(False Positive = unnecessary workshop inspection; False Negative = missed component breakdown resulting in highway failure).*

---

## Repository Directory Architecture

```text
DS605_Scania_APS_Project/
├── data/
│   ├── raw/
│   │   ├── aps_failure_training_set.csv
│   │   └── aps_failure_test_set.csv
│   └── processed_scania_data.npz
├── plots/
│   ├── 01_class_imbalance.png
│   ├── 02_missingness_profile.png
│   ├── 03_sensor_distributions.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_pca_scree_plot.png
│   ├── 06_bias_variance_tradeoff.png
│   ├── 07_cost_threshold_curve.png
│   ├── 08_roc_and_pr_curves.png
│   ├── 09_failure_dendrogram.png
│   └── 10_clusters_2d_pca.png
├── notebooks/
│   ├── 01_Data_Preprocessing_EDA_PCA_202618047.ipynb
│   └── 02_Modeling_Imbalance_Cost_Clustering_202618052.ipynb
└── README.md
```

---

## Task Distribution & Methodology

### Part 1: Pre-Modeling Pipeline (`202618047`)
* **Data Ingestion & Target Encoding:** Parsing missing values (`na` strings mapped to `NaN`) and converting class labels to binary integers.
* **Exploratory Data Analysis (Unit 2):** Analyzing central tendency, dispersion, and class distribution across high-dimensional sensor data.
* **Feature Engineering:** Constructing ECU dropout indicators (`fe_missing_count`, `fe_missing_ratio`) and operational histogram family aggregations (row sums and maximums across 7 multi-bin groups).
* **Data Cleaning & Preprocessing (Unit 3):** Pruning columns with $>70\%$ missingness, removing zero-variance constant features via `VarianceThreshold`, applying median imputation, and scaling via `RobustScaler`.
* **Dimensionality Reduction (Unit 6):** Applying Principal Component Analysis (PCA) and TruncatedSVD, generating scree plots, and preserving 95% cumulative explained variance.

### Part 2: Modeling, Optimization & Clustering (`202618052`)
* **Class Imbalance Resolution (Unit 5):** Benchmarking baseline models against Random Undersampling, Random Oversampling, SMOTE, and cost-sensitive class weighting.
* **Supervised Model Benchmarking (Units 4, 5 & 6):** Implementing and tuning 12 machine learning models:
  * Lazy Learner: $k$-Nearest Neighbors ($k$-NN)
  * Linear Models: Ridge Logistic Regression and ElasticNet SGD Classifier
  * Support Vector Machines: Linear Hinge-Loss SVM and RBF Kernel SVM
  * Decision Trees: CART Decision Trees with Bias-Variance depth analysis
  * Ensemble Methods: Bagging, Pasting, Random Forest, AdaBoost, HistGradientBoosting, and Soft Voting Classifier
* **Bayesian Cost-Matrix Threshold Optimization (Unit 4):** Sweeping decision probability cutoffs to determine the optimal threshold $t^*$ minimizing the official Scania cost formula.
* **Unsupervised Failure-Regime Analysis (Unit 7):** Applying $K$-Means clustering, Ward linkage hierarchical clustering (dendrogram analysis), and DBSCAN to isolate distinct mechanical failure modes among failed trucks.

---

## Getting Started & Execution Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/scania-aps-predictive-maintenance.git
   cd scania-aps-predictive-maintenance
   ```
2. Install required Python dependencies:
   ```bash
   pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn scipy joblib
   ```
3. Download the raw dataset CSV files into `data/raw/`.
4. Execute `notebooks/01_Data_Preprocessing_EDA_PCA_202618047.ipynb` to generate the preprocessed handoff file (`data/processed_scania_data.npz`).
5. Run `notebooks/02_Modeling_Imbalance_Cost_Clustering_202618052.ipynb` to execute model training, threshold tuning, and clustering analyses.