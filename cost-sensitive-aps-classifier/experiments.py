"""Course experiments on development data only; never changes the deployed model.

Uses a log1p + robust scaling PCA branch to investigate extreme-value dominance.
These are exploratory validation comparisons, not independent final test claims.
"""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (silhouette_score, davies_bouldin_score, f1_score,
                             roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from aps_pipeline import FeatureEngineer, MissingnessPruner
from metrics import report, select_threshold
from train import load_uci, build_pipeline


def steps():
    return [("features", FeatureEngineer()), ("missingness", MissingnessPruner()),
            ("imputer", SimpleImputer(strategy="median")), ("variance", VarianceThreshold())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("experiments"))
    a = p.parse_args()
    if a.out.exists(): p.error("Choose a fresh experiment output directory.")
    X, y = load_uci(a.train)
    if (X < 0).any().any(): raise ValueError("log1p branch requires verified nonnegative counters.")
    fit, val, yf, yv = train_test_split(X, y, test_size=.15, stratify=y, random_state=42)
    a.out.mkdir(parents=True)
    def log_pca():
        # Each experiment owns its own fitted transformers.
        return [("log", FunctionTransformer(np.log1p)), ("scale", RobustScaler()),
                ("pca", PCA(n_components=50, random_state=42))]
    def lr(weight):
        return LogisticRegression(class_weight=weight, solver="lbfgs", max_iter=3000, random_state=42)
    models = {
        "RF balanced": build_pipeline(),
        "HGB balanced": Pipeline(steps()+[("model", HistGradientBoostingClassifier(class_weight="balanced", random_state=42))]),
        "LR balanced log-PCA": Pipeline(steps()+log_pca()+[("model", lr("balanced"))]),
        "LR SMOTE log-PCA": Pipeline(steps()+log_pca()+[("sample", SMOTE(sampling_strategy=.2, random_state=42)), ("model", lr(None))]),
        "LR undersampled log-PCA": Pipeline(steps()+log_pca()+[("sample", RandomUnderSampler(sampling_strategy=.5, random_state=42)), ("model", lr(None))]),
    }
    results, fitted = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for name, model in models.items():
            print("Fitting", name, flush=True)
            model.fit(fit, yf)
            score = model.predict_proba(val)[:, 1]
            t, _ = select_threshold(yv, score)
            results.append(dict(model=name, **report(yv, score, t)))
            fitted[name] = model
    pd.DataFrame(results).to_csv(a.out / "validation_benchmarks.csv", index=False)
    # Bias/complexity illustration on the same fit/validation partitions.
    clean = Pipeline(steps()).fit(fit, yf)
    zf, zv = clean.transform(fit), clean.transform(val)
    complexity = []
    for depth in [2, 4, 6, 8, 10, 12, 16, 20, 25]:
        tree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=42).fit(zf, yf)
        complexity.append(dict(depth=depth, train_f1=f1_score(yf, tree.predict(zf)),
                               validation_f1=f1_score(yv, tree.predict(zv))))
    complexity = pd.DataFrame(complexity)
    complexity.to_csv(a.out / "complexity.csv", index=False)
    complexity.set_index("depth").plot(marker="o"); plt.ylabel("Minority F1"); plt.tight_layout()
    plt.savefig(a.out / "complexity.png", dpi=150); plt.close()
    # PCA/clustering fit only on the fit partition; failures remain anonymous groups.
    projection = Pipeline(steps()+log_pca()).fit(fit, yf)
    failures = projection.transform(fit.loc[yf == 1])[:, :15]
    cluster_results = []
    for k in [2, 3, 4]:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(failures)
        sizes = np.bincount(km.labels_).tolist()
        cluster_results.append(dict(k=k, silhouette=float(silhouette_score(failures, km.labels_)),
                                    davies_bouldin=float(davies_bouldin_score(failures, km.labels_)), sizes=sizes))
    (a.out / "clusters.json").write_text(json.dumps(cluster_results, indent=2))
    selected = max(cluster_results, key=lambda row: row["silhouette"])["k"]
    km = KMeans(n_clusters=selected, n_init=10, random_state=42).fit(failures)
    plt.scatter(failures[:, 0], failures[:, 1], c=km.labels_, s=10, alpha=.6)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"Exploratory failure groups (k={selected})")
    plt.tight_layout(); plt.savefig(a.out / "clusters.png", dpi=150); plt.close()
    rng = np.random.default_rng(42)
    subset = failures[rng.choice(len(failures), min(250, len(failures)), replace=False)]
    dendrogram(linkage(subset, method="ward"), truncate_mode="lastp", p=12)
    plt.title("Ward linkage — seeded subset of fit-partition failures")
    plt.tight_layout(); plt.savefig(a.out / "dendrogram.png", dpi=150); plt.close()
    pca = projection.named_steps["pca"]
    plt.plot(np.arange(1, 51), np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Components"); plt.ylabel("Cumulative explained variance")
    plt.tight_layout(); plt.savefig(a.out / "pca.png", dpi=150); plt.close()
    rf = fitted["RF balanced"].predict_proba(val)[:, 1]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(yv, rf); precision, recall, _ = precision_recall_curve(yv, rf)
    ax[0].plot(fpr, tpr); ax[0].set(xlabel="False positive rate", ylabel="Recall", title="Validation ROC")
    ax[1].plot(recall, precision); ax[1].axhline(yv.mean(), linestyle="--")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Validation PR")
    fig.tight_layout(); fig.savefig(a.out / "roc_pr.png", dpi=150); plt.close(fig)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__": main()
