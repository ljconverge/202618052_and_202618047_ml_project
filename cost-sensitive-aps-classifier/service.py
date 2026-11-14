"""Testable upload, release integrity and scoring boundary. No Streamlit imports."""
import csv
import hashlib
from io import BytesIO, StringIO
import importlib.metadata
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from aps_pipeline import validate_raw

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 10000
RELEASE_FILES = ("aps_pipeline.joblib", "metadata.json", "validation_predictions.csv",
                 "validation_curve.csv")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_fingerprint(root):
    return digest(Path(root) / "manifest.json")


def load_release(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != set(RELEASE_FILES):
        raise ValueError("Incomplete release manifest.")
    for name in RELEASE_FILES:
        if digest(root / name) != manifest[name]:
            raise ValueError(f"Release checksum mismatch: {name}")
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1:
        raise ValueError("Unsupported release schema.")
    if len(meta["raw_features"]) != 170 or len(set(meta["raw_features"])) != 170:
        raise ValueError("Invalid raw feature schema.")
    if not 0 <= meta["threshold"] <= np.nextafter(1., 2.):
        raise ValueError("Invalid release threshold.")
    if meta["python_major_minor"] != list(sys.version_info[:2]):
        raise ValueError("Python major/minor differs from the training environment.")
    for package, version in meta["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"{package} differs from the training environment.")
    if meta["source_sha256"] != digest(Path(__file__).with_name("aps_pipeline.py")):
        raise ValueError("Transformer source differs from the training release.")
    # A checksum detects corruption, not a malicious artifact + manifest pair.
    # Deployment operator must trust and protect the entire release directory.
    model = joblib.load(root / "aps_pipeline.joblib")
    if list(model.named_steps["features"].feature_names_in_) != meta["raw_features"]:
        raise ValueError("Model and metadata feature schemas disagree.")
    if model.classes_.tolist() != [0, 1]:
        raise ValueError("Model does not have the expected class mapping.")
    model.named_steps["model"].n_jobs = 1  # bound per-request CPU use
    evidence = pd.read_csv(root / "validation_predictions.csv")
    from metrics import cost_row
    if set(evidence.columns) != {"label", "score"}:
        raise ValueError("Invalid validation evidence schema.")
    cost_row(evidence.label, evidence.score, meta["threshold"])
    return model, meta, evidence


def parse_upload(payload, features):
    if len(payload) > MAX_BYTES:
        raise ValueError("File exceeds 10 MB.")
    if not payload:
        raise ValueError("The uploaded file is empty.")
    try:
        text = payload.decode("utf-8-sig")
        reader = csv.reader(StringIO(text), strict=True)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError("Duplicate CSV column names are not allowed.")
        expected = set(features)
        if set(header) not in (expected, expected | {"record_id"}):
            raise ValueError("CSV must contain the 170 raw sensor columns and optional record_id only.")
        count = 0
        for row in reader:
            if not row:  # allow blank lines, as pandas does
                continue
            count += 1
            if len(row) != len(header):
                raise ValueError(f"CSV record {count} has the wrong number of fields.")
            if count > MAX_ROWS:
                raise ValueError("At most 10,000 records per batch.")
        frame = pd.read_csv(StringIO(text), na_values=["na", ""],
                            keep_default_na=False, dtype={"record_id": "string"})
    except (UnicodeError, csv.Error, StopIteration, pd.errors.ParserError,
            pd.errors.EmptyDataError) as exc:
        raise ValueError("Invalid CSV. Use UTF-8 with one header row and comma-separated fields.") from exc
    ids = (frame.pop("record_id") if "record_id" in frame else
           pd.Series([f"row-{i+1}" for i in range(len(frame))], dtype="string"))
    ids = ids.astype("string").str.strip()
    if ids.isna().any() or (ids == "").any() or not ids.is_unique:
        raise ValueError("record_id must be nonblank and unique for every record.")
    if ids.str.len().gt(128).any() or ids.str.contains(r"[\x00-\x1f\x7f]", regex=True).any():
        raise ValueError("record_id must be at most 128 characters without control characters.")
    raw = validate_raw(frame, features)
    return raw, ids.reset_index(drop=True)


def score_batch(model, meta, raw, ids):
    scores = model.predict_proba(raw)[:, 1]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model returned invalid scores.")
    result = pd.DataFrame({"record_id": ids.astype(str).to_numpy(), "model_score": scores,
                           "missing_fraction": raw.isna().mean(axis=1).to_numpy()})
    result["action"] = np.where(scores >= meta["threshold"], "Review for inspection", "Below threshold")
    result["model_id"] = meta["model_id"]
    result["threshold"] = meta["threshold"]
    return result


def export_csv(frame):
    safe = frame.copy()
    # Defend spreadsheet users against formulas in untrusted identifiers.
    for name in safe.select_dtypes(include=["object", "string"]).columns:
        safe[name] = safe[name].map(lambda x: "'" + x if isinstance(x, str)
                                   and x.lstrip().startswith(("=", "+", "-", "@")) else x)
    return safe.to_csv(index=False).encode("utf-8")


def is_allowed_user(user, allowed):
    email = str(user.get("email", "")).strip().lower()
    return bool(user.get("is_logged_in") and user.get("email_verified") is True
                and email and email in {str(x).strip().lower() for x in allowed})
