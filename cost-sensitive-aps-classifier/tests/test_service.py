from io import BytesIO
import json
import numpy as np
import pandas as pd
import pytest
from metrics import cost_row, select_threshold, threshold_curve
from service import (MAX_BYTES, export_csv, is_allowed_user, load_release,
                     parse_upload, score_batch, digest)


def test_upload_reorders_and_preserves_ids(raw, features):
    raw.insert(0, "record_id", [f"{i:04d}" for i in range(len(raw))])
    parsed, ids = parse_upload(raw[raw.columns[::-1]].to_csv(index=False).encode(), features)
    assert list(parsed.columns) == features
    assert ids.iloc[0] == "0000"


@pytest.mark.parametrize("kind", ["empty", "bytes", "duplicate_header", "short_row", "extra_column",
                                         "bad_numeric", "infinite", "overflow", "all_missing",
                                         "duplicate_id", "blank_id", "control_id", "long_id", "too_many"])
def test_bad_upload_rejected(raw, features, kind):
    if kind == "empty": payload = b""
    elif kind == "bytes": payload = b"\xff\xfe"
    elif kind == "duplicate_header": payload = (",".join(features + [features[0]]) + "\n").encode()
    elif kind == "short_row": payload = (",".join(features) + "\n1,2\n").encode()
    elif kind == "extra_column": payload = raw.assign(class_label=0).to_csv(index=False).encode()
    else:
        if kind == "bad_numeric": raw = raw.astype(object); raw.iloc[0, 0] = "bad"
        elif kind == "infinite": raw.iloc[0, 0] = np.inf
        elif kind == "overflow": raw.iloc[0, 0] = 1e100
        elif kind == "all_missing": raw.iloc[0, :] = np.nan
        elif kind == "duplicate_id": raw["record_id"] = "x"
        elif kind == "blank_id": raw["record_id"] = [" "] + list(map(str, range(1, len(raw))))
        elif kind == "control_id": raw["record_id"] = ["a\tb"] + list(map(str, range(1, len(raw))))
        elif kind == "long_id": raw["record_id"] = ["a" * 129] + list(map(str, range(1, len(raw))))
        elif kind == "too_many": raw = pd.concat([raw.iloc[[0]]] * 10001, ignore_index=True)
        payload = raw.to_csv(index=False).encode()
    with pytest.raises(ValueError): parse_upload(payload, features)


def test_size_cap(features):
    with pytest.raises(ValueError, match="10 MB"):
        parse_upload(b"x" * (MAX_BYTES + 1), features)


def test_all_missing_histogram_valid(raw, features):
    raw.loc[0, [c for c in raw if c.startswith("ag_")]] = np.nan
    parsed, _ = parse_upload(raw.to_csv(index=False).encode(), features)
    assert parsed.iloc[0].isna().sum() == 10


def test_export_defends_formulas():
    encoded = export_csv(pd.DataFrame({"record_id": ["=1+1", " +CMD", "@SUM(1)", "normal"], "score": [.1]*4}))
    out = pd.read_csv(BytesIO(encoded))
    assert out.record_id.tolist() == ["'=1+1", "' +CMD", "'@SUM(1)", "normal"]


def test_release_and_scoring_parity(release, raw):
    model, meta, evidence = load_release(release)
    ids = pd.Series([f"row-{i}" for i in range(len(raw))])
    result = score_batch(model, meta, raw, ids)
    assert len(result) == len(raw)
    assert np.isfinite(result.model_score).all()
    assert result.model_score.iloc[0] == model.predict_proba(raw.iloc[[0]])[0, 1]
    assert set(result.action) <= {"Review for inspection", "Below threshold"}


@pytest.mark.parametrize("name", ["aps_pipeline.joblib", "metadata.json", "validation_predictions.csv", "validation_curve.csv"])
def test_corrupt_release_rejected(release, name):
    with (release / name).open("ab") as handle: handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"): load_release(release)


def test_environment_mismatch_rejected(release):
    path = release / "metadata.json"
    meta = json.loads(path.read_text()); meta["versions"]["scikit-learn"] = "0.0.0"
    path.write_text(json.dumps(meta))
    manifest = json.loads((release / "manifest.json").read_text()); manifest["metadata.json"] = digest(path)
    (release / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="environment"): load_release(release)


@pytest.mark.parametrize("user,allowed,expected", [
    ({"is_logged_in": True, "email_verified": True, "email": "A@Example.com"}, ["a@example.com"], True),
    ({"is_logged_in": False, "email_verified": True, "email": "a@example.com"}, ["a@example.com"], False),
    ({"is_logged_in": True, "email": "a@example.com"}, ["a@example.com"], False),
    ({"is_logged_in": True, "email_verified": True, "email": "b@example.com"}, ["a@example.com"], False),
    ({"is_logged_in": True, "email_verified": True, "email": "a@example.com"}, [], False),
])
def test_authorization(user, allowed, expected):
    assert is_allowed_user(user, allowed) is expected


def test_exact_curve_matches_brute_force():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 150)
    scores = rng.choice([0., .1, .2, .8, 1.], 150)
    t, curve = select_threshold(y, scores)
    for row in curve.to_dict("records"):
        assert row == cost_row(y, scores, row["threshold"])
    assert cost_row(y, scores, t)["cost"] == curve.cost.min()


@pytest.mark.parametrize("labels,scores", [([], []), ([2], [.2]), ([.5], [.2]), ([0], [np.nan]), ([0], [1.1])])
def test_invalid_metric_inputs(labels, scores):
    with pytest.raises(ValueError): threshold_curve(labels, scores)
