# FraudLens: Does Side Information Improve Financial Statement Fraud Detection?

**Course:** DS605 · Fundamentals of Machine Learning, M.Sc. Data Science, DA-IICT
**Domain:** Finance

**Team**

| Roll number |
|---|
| 202618052 |
| 202618047 |

---

## Problem Statement

Financial Statement Fraud Detection: Evaluating the Added Value of Macroeconomic and News-Sentiment Context Using Classical Machine Learning.

Most fraud detection uses only a company's own accounting ratios. This project tests whether adding side information (the macroeconomic environment and the tone of news about the company) improves detection over ratios alone.

## Approach

- **Unit:** one company in one fiscal year.
- **Target:** `fraud_label` = 1 if the company-year falls within an SEC-enforced accounting-fraud case, else 0.
- **Run A:** core accounting ratios only.
- **Run B:** core ratios plus side information (`macro_*`, `news_*`).
- **Models:** scikit-learn classifiers (kNN, logistic regression, SVM, decision tree, random forest, gradient boosting, voting).
- **Validation:** grouped, stratified cross-validation, so no company appears in both training and test folds.
- **Metric:** PR-AUC with recall, precision and F1, reported as mean ± standard deviation.

## Data Sources

| Source | Used for |
|---|---|
| SEC EDGAR XBRL company facts | Core financial ratios |
| SEC AAER enforcement releases | Fraud labels |
| World Bank Indicators API | Macro side information |
| GDELT DOC 2.0 API (via Scrapy) | News side information (VADER sentiment) |
| UCI credit datasets | Benchmark only |

## Repository Structure

```
├── config/config.yaml        project settings
├── data/
│   ├── raw/                  unchanged downloads (not tracked)
│   ├── interim/              cleaned files per source (not tracked)
│   └── processed/            master dataset and feature sets
├── docs/                     data dictionary
├── src/
│   ├── acquisition/          data download and news spider
│   ├── cleaning/             cleaning per source
│   ├── merge/                master table
│   ├── audit/                raw-data audit and contract check
│   ├── features/             Run A and Run B features
│   ├── models/               splits, metrics, models
│   └── experiments/          ablation study
├── reports/                  figures and tables
├── requirements.txt
└── README.md
```

## Setup

Python 3.12 in a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python check_env.py
```

## Progress Log

### Day 1: Repository setup and data audit

- [ ] Repository created and README committed
- [ ] Environment set up on both machines (`check_env.py` passes)
- [ ] Raw data audit run (`src/audit/audit_raw_data.py`)
- [ ] Distinct fraud companies counted
- [ ] Row grain identified
- [ ] World Bank country set to US in `config/config.yaml`
- [ ] Data dictionary v0 started (`docs/data_dictionary.md`)

**Findings**
- Rows and columns in the master file: *to be filled*
- Distinct fraud companies: *to be filled*
- Row grain: *to be filled*

**Blockers:** none recorded