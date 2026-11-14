import importlib.metadata
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
from train import build_pipeline
from service import RELEASE_FILES, digest


@pytest.fixture
def features():
    return ["aa_000"] + [f"ag_{i:03d}" for i in range(10)] + [f"sensor_{i:03d}" for i in range(159)]


@pytest.fixture
def raw(features):
    return pd.DataFrame(np.random.default_rng(42).uniform(0, 100, (30, 170)), columns=features)


@pytest.fixture
def release(tmp_path, raw, features):
    import aps_pipeline
    root = tmp_path / "release"
    root.mkdir()
    model = build_pipeline(n_estimators=5).fit(raw, [0] * 20 + [1] * 10)
    joblib.dump(model, root / "aps_pipeline.joblib")
    meta = dict(schema_version=1, model_id="test-release", threshold=.2,
                raw_features=features, python_major_minor=list(sys.version_info[:2]),
                source_sha256=digest(aps_pipeline.__file__),
                versions={p: importlib.metadata.version(p) for p in
                          ("numpy", "pandas", "scipy", "scikit-learn", "joblib")},
                created_utc="2026-11-14T00:00:00+00:00", metrics={"test": {"cost": 0}})
    (root / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    pd.DataFrame({"label": [0, 1], "score": [.1, .8]}).to_csv(root / "validation_predictions.csv", index=False)
    pd.DataFrame({"threshold": [.2], "cost": [0]}).to_csv(root / "validation_curve.csv", index=False)
    (root / "manifest.json").write_text(json.dumps({n: digest(root / n) for n in RELEASE_FILES}), encoding="utf-8")
    return root
