"""Leakage-safe training; publish a complete immutable release atomically."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from aps_pipeline import FeatureEngineer, MissingnessPruner, validate_raw
from metrics import cost_row, report, select_threshold
from service import RELEASE_FILES, digest


def load_uci(path):
    frame = pd.read_csv(path, skiprows=20, na_values="na")
    if "class" not in frame:
        raise ValueError("Expected UCI CSV with class column after 20 metadata rows.")
    y = frame.pop("class").map({"neg": 0, "pos": 1})
    if y.isna().any() or y.nunique() != 2:
        raise ValueError("Expected both neg and pos labels, with no unknown labels.")
    if frame.shape[1] != 170:
        raise ValueError("Expected 170 raw sensor columns.")
    return validate_raw(frame, frame.columns.tolist()), y.astype(int)


def build_pipeline(n_estimators=100):
    return Pipeline([
        ("features", FeatureEngineer()), ("missingness", MissingnessPruner()),
        ("imputer", SimpleImputer(strategy="median")), ("variance", VarianceThreshold()),
        ("scaler", RobustScaler()),
        ("model", RandomForestClassifier(n_estimators=n_estimators,
             class_weight="balanced_subsample", random_state=42, n_jobs=2)),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; use a new release directory.")
    X, y = load_uci(args.train)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    print("Fitting full pipeline on", len(X_fit), "rows", flush=True)
    model = build_pipeline()
    model.fit(X_fit, y_fit)
    val_scores = model.predict_proba(X_val)[:, 1]
    threshold, curve = select_threshold(y_val, val_scores)
    print("Validation threshold fixed:", threshold, flush=True)
    # Preserve this exact fitted model; a full-data refit changes its scores.
    X_test, y_test = load_uci(args.test)
    X_test = validate_raw(X_test, X.columns.tolist())
    test_scores = model.predict_proba(X_test)[:, 1]
    metrics = {
        "validation_selected": report(y_val, val_scores, threshold),
        "test_selected": report(y_test, test_scores, threshold),
        "test_default_050": report(y_test, test_scores, 0.5),
        "test_always_negative": cost_row(y_test, test_scores, np.nextafter(1., 2.)),
        "test_always_positive": cost_row(y_test, test_scores, 0.),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".aps-staging-", dir=args.out.parent))
    try:
        joblib.dump(model, stage / "aps_pipeline.joblib", compress=3)
        meta = {
            "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": args.out.name, "threshold": threshold,
            "score_kind": "uncalibrated model score", "raw_features": X.columns.tolist(),
            "fit_rows": len(X_fit), "validation_rows": len(X_val), "test_rows": len(X_test),
            "fit_positive": int(y_fit.sum()), "validation_positive": int(y_val.sum()),
            "false_positive_cost": 10, "false_negative_cost": 500,
            "source_sha256": digest(Path(__file__).with_name("aps_pipeline.py")),
            "training_data_sha256": digest(args.train), "test_data_sha256": digest(args.test),
            "fit_index_sha256": hashlib.sha256(X_fit.index.to_numpy().tobytes()).hexdigest(),
            "validation_index_sha256": hashlib.sha256(X_val.index.to_numpy().tobytes()).hexdigest(),
            "python": platform.python_version(), "python_major_minor": list(sys.version_info[:2]),
            "versions": {p: importlib.metadata.version(p) for p in
                         ("numpy", "pandas", "scipy", "scikit-learn", "joblib")},
            "seed": 42, "model_parameters": model.named_steps["model"].get_params(),
            "metrics": metrics,
        }
        (stage / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        curve.to_csv(stage / "validation_curve.csv", index=False)
        pd.DataFrame({"label": y_val.to_numpy(), "score": val_scores}).to_csv(
            stage / "validation_predictions.csv", index=False)
        manifest = {name: digest(stage / name) for name in RELEASE_FILES}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        from service import load_release
        restored, _, _ = load_release(stage)
        np.testing.assert_allclose(restored.predict_proba(X_test.iloc[:32])[:, 1], test_scores[:32])
        stage.rename(args.out)
    finally:
        if stage.exists():
            if stage.resolve().parent != args.out.parent.resolve() or not stage.name.startswith(".aps-staging-"):
                raise RuntimeError("Unexpected staging directory")
            shutil.rmtree(stage)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
