"""Small regression checks; run with python -m unittest -v test_pipeline.py."""
import tempfile
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from aps_pipeline import FeatureEngineer, MissingnessPruner, validate_raw
from train import cost_row


class PipelineChecks(unittest.TestCase):
    def test_schema_and_invalid_input(self):
        X = pd.DataFrame({"aa_000": [1], "ag_000": [2]})
        pd.testing.assert_frame_equal(
            validate_raw(X[["ag_000", "aa_000"]], X.columns.tolist()), X)
        for bad in [X.drop(columns="aa_000"), X.assign(unknown=2),
                    X.assign(aa_000="invalid"), X.assign(aa_000=np.inf)]:
            with self.assertRaises(ValueError):
                validate_raw(bad, X.columns.tolist())

    def test_missing_histogram_is_unknown(self):
        X = pd.DataFrame({"aa_000": [1., 2.], "ag_000": [np.nan, 4.],
                          "ag_001": [np.nan, 6.]})
        out = FeatureEngineer().fit_transform(X)
        self.assertTrue(np.isnan(out.loc[0, "fe_ag_sum"]))
        self.assertEqual(out.loc[1, "fe_ag_sum"], 10.)

    def test_validation_does_not_change_pruning_or_medians(self):
        train = pd.DataFrame({"a": [1., 3., 5., np.nan],
                              "b": [np.nan, np.nan, np.nan, 4.]})
        prune = MissingnessPruner().fit(train)
        imp = SimpleImputer(strategy="median").fit(prune.transform(train))
        before = imp.statistics_.copy()
        val = pd.DataFrame({"a": [np.nan], "b": [10000.]})
        self.assertEqual(imp.transform(prune.transform(val))[0, 0], 3.)
        np.testing.assert_array_equal(before, imp.statistics_)
        self.assertEqual(prune.columns_, ["a"])

    def test_serialization_and_batch_single_parity(self):
        X = pd.DataFrame({"aa_000": [1., 2., 3., 4., 5., 6.],
                          "ag_000": [0., 0., 1., 1., 2., 3.]})
        model = Pipeline([("features", FeatureEngineer()),
                          ("imputer", SimpleImputer()),
                          ("model", RandomForestClassifier(n_estimators=5, random_state=42))])
        model.fit(X, [0, 0, 0, 1, 1, 1])
        expected = model.predict_proba(X)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_allclose(expected, restored.predict_proba(X))
            np.testing.assert_allclose(expected[0], restored.predict_proba(X.iloc[[0]])[0])

    def test_cost_and_single_class(self):
        self.assertEqual(cost_row([0, 1], [.8, .2], .5)["cost"], 510)
        self.assertEqual(cost_row([0, 0], [.1, .2], .5)["cost"], 0)


if __name__ == "__main__":
    unittest.main()
