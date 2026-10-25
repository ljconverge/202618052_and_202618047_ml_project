# 202618052_and_202618047_ml_project

**FraudLens: Does Side Information Improve Financial Statement Fraud Detection?**

Course project for **DS605 · Fundamentals of Machine Learning**, M.Sc. Data Science, DA-IICT.

| Roll number | Team |
|---|---|
| 202618052 | Lakshita Pagaria |
| 202618047 | Ameya Trivedi |

## Repository Contents

| Path | Description |
|---|---|
| [`FraudLens/`](FraudLens/) | Complete project: code, data pipeline, configuration, and documentation |
| [`FraudLens/README.md`](FraudLens/README.md) | Detailed project README: problem statement, approach, setup, and daily progress log |

## Project Summary

Most financial fraud detection relies only on a company's own accounting ratios. This project tests whether adding **side information** (macroeconomic conditions and the tone of news about the company) improves detection.

- **Unit:** one company in one fiscal year
- **Target:** enforced accounting fraud (SEC AAER releases)
- **Comparison:** Run A (accounting ratios only) against Run B (ratios plus side information)
- **Methods:** classical machine learning with scikit-learn, leakage-safe grouped cross-validation, PR-AUC as the primary metric

## Getting Started

All commands are run from inside the `FraudLens/` folder.

```bash
cd FraudLens
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python check_env.py
```

See [`FraudLens/README.md`](FraudLens/README.md) for the full run order and the progress log.

## Data Sources

SEC EDGAR, SEC AAER enforcement releases, World Bank Indicators API, GDELT DOC 2.0 API, and UCI credit datasets (benchmark only). Details are in the project README.