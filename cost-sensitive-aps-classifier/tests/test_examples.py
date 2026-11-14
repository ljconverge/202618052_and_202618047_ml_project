from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
from service import load_release, parse_upload, score_batch, export_csv

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_example_matches_the_shipped_release():
    model, meta, _ = load_release(ROOT / 'artifacts')
    raw, ids = parse_upload((ROOT / 'examples/aps_sample_input.csv').read_bytes(), meta['raw_features'])
    actual = score_batch(model, meta, raw, ids)
    expected = pd.read_csv(ROOT / 'examples/expected_sample_scores.csv')
    assert actual.record_id.tolist() == expected.record_id.tolist()
    assert actual.action.tolist() == expected.action.tolist()
    np.testing.assert_allclose(actual.model_score, expected.model_score, rtol=1e-12, atol=1e-12)
    exported = pd.read_csv(BytesIO(export_csv(actual)))
    np.testing.assert_allclose(exported.model_score, expected.model_score, rtol=1e-12, atol=1e-12)
