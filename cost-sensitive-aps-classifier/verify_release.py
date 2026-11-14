"""Full-data integration verification and latency measurements, separate from training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from metrics import report
from service import load_release, parse_upload, score_batch, export_csv
from train import load_uci


def verify(root, test_path):
    model, meta, _ = load_release(root)
    X, y = load_uci(test_path)
    scores = model.predict_proba(X)[:, 1]
    actual = report(y, scores, meta["threshold"])
    for key, value in meta["metrics"]["test_selected"].items():
        np.testing.assert_allclose(actual[key], value, rtol=1e-12, atol=1e-12)
    payload = X.iloc[:100].to_csv(index=False, na_rep="na").encode()
    raw, ids = parse_upload(payload, meta["raw_features"])
    result = score_batch(model, meta, raw, ids)
    np.testing.assert_allclose(result.model_score, scores[:100])
    assert len(export_csv(result)) > 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(lambda _: score_batch(model, meta, raw, ids).model_score.to_numpy(), range(4)))
    for output in concurrent: np.testing.assert_array_equal(output, scores[:100])
    timings = {}
    for size in [1, 100, 1000, 10000]:
        payload = X.iloc[:size].to_csv(index=False, na_rep="na").encode()
        elapsed = []
        for _ in range(3):
            start = time.perf_counter()
            raw, ids = parse_upload(payload, meta["raw_features"])
            out = score_batch(model, meta, raw, ids)
            assert len(out) == size
            elapsed.append((time.perf_counter() - start) * 1000)
        timings[str(size)] = dict(csv_bytes=len(payload), median_ms=float(np.median(elapsed)),
                                  maximum_ms=float(max(elapsed)), repetitions=3)
    # Paired ordinary bootstrap: uncertainty is conditional on this benchmark
    # and fixed model; not uncertainty about future fleet/domain shifts.
    labels = y.to_numpy()
    chosen = scores >= meta["threshold"]
    baseline = scores >= .5
    losses = np.where((labels == 1) & ~chosen, 500, np.where((labels == 0) & chosen, 10, 0))
    base_loss = np.where((labels == 1) & ~baseline, 500, np.where((labels == 0) & baseline, 10, 0))
    rng = np.random.default_rng(42)
    costs, savings = [], []
    for _ in range(1000):
        idx = rng.integers(0, len(labels), len(labels))
        costs.append(int(losses[idx].sum()))
        savings.append(int((base_loss[idx] - losses[idx]).sum()))
    return dict(status="passed", full_test_rows=len(X), metrics=actual,
                integration_checks=["release integrity", "full-test metric reproduction", "raw CSV scoring parity",
                                    "CSV export", "four concurrent scoring calls", "10,000-row batch"],
                latency=timings, cost_bootstrap_95_percent=np.percentile(costs, [2.5, 97.5]).tolist(),
                saving_vs_050_bootstrap_95_percent=np.percentile(savings, [2.5, 97.5]).tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("verification.json"))
    a = p.parse_args()
    evidence = verify(a.artifacts, a.test)
    a.out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
