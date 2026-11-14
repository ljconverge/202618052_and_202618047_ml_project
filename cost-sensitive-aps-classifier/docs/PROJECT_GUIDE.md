# Complete project guide: changes, migration, notebook and verification

Prepared 14 November 2026 for Lakshita Pagaria (202618052) and Ameya Trivedi
(202618047), DS605: Fundamentals of Machine Learning.

## 1. What was actually changed and tested

Yes: the earlier upgrade replaced the application and corrected the shared
training/deployment pipeline. It did **not** edit the original project in your
PyCharm directory, and it did not update your old notebook. This final delivery
adds a new matching notebook and a single coherent repository layout.

The eight runtime modules (`app.py`, `launch.py`, `train.py`, `experiments.py`,
`verify_release.py`, `aps_pipeline.py`, `service.py`, `metrics.py`) are **byte-for-byte
unchanged from the previously tested production-release package**. Their hashes
are recorded in `reports/runtime-code-baseline.json`. You do not need to manually
edit their imports or paste replacement code into 24 old notebook cells.

This final consolidation changes only organization/documentation, test discovery,
example data, notebook tooling and the new notebook. It adds an example-to-model
integration test. The new notebook uses the same modules rather than maintaining
a second training implementation.

A supplemental advisory check of research/test/notebook packages also identified
an issue in pytest 8.3.5. The development lock now uses pytest 9.0.3 with its
Pygments dependency; the suite is rerun with that updated test runner. The runtime
model, scientific packages and eight serving/training modules are unaffected.

Testing is substantial but is not literally “everything.” The code, training,
raw-data scoring, automated UI states and local HTTP startup were tested. Live
OIDC credentials, Docker/Linux hosting, TLS/reverse-proxy configuration and visual
browser/mobile acceptance still require checks on your actual host. A benchmark
does not itself validate prospective live fleet decisions.

## 2. Final repository structure

Use this layout exactly as delivered:

```text
cost-sensitive-aps-classifier/          # repository root; run commands here
├── .github/
│   └── workflows/checks.yml           # CI test definition
├── .streamlit/
│   ├── config.toml                    # theme, limits, server settings
│   └── secrets.toml.example           # placeholders only; real file is ignored
├── artifacts/                        # one approved, complete model release
│   ├── aps_pipeline.joblib
│   ├── metadata.json
│   ├── manifest.json
│   ├── validation_curve.csv
│   └── validation_predictions.csv
├── dataset/
│   ├── README.md
│   └── raw_data/
│       ├── aps_failure_training_set.csv
│       └── aps_failure_test_set.csv
├── docs/
│   └── PROJECT_GUIDE.md               # this one explanation/migration guide
├── examples/
│   ├── README.md
│   ├── aps_sample_input.csv
│   └── expected_sample_scores.csv
├── experiment-results/               # generated research evidence, used by UI
│   ├── validation_benchmarks.csv
│   ├── complexity.csv
│   ├── clusters.json
│   └── *.png                         # PCA, ROC/PR, complexity, clustering, Ward
├── notebooks/
│   ├── README.md
│   └── 01_scania_aps_end_to_end.ipynb  # use this notebook; Run All
├── reports/
│   ├── test-results.xml
│   ├── pytest-output.log
│   ├── verification.json
│   ├── notebook-execution.json
│   ├── dependency-audit.json
│   └── runtime-code-baseline.json
├── tests/
│   ├── conftest.py
│   ├── test_pipeline.py
│   ├── test_service.py
│   ├── test_app.py
│   ├── test_launch.py
│   └── test_examples.py
├── app.py                            # serving UI
├── aps_pipeline.py                   # importable preprocessing transformers
├── service.py                        # parsing, release loading, scoring/export
├── metrics.py                        # shared loss and threshold logic
├── train.py                          # train + publish immutable release
├── experiments.py                    # development-only course experiments
├── verify_release.py                 # real-data integration checks
├── launch.py                         # startup preflight + server
├── requirements.txt                  # exact runtime versions
├── requirements-dev.txt              # testing/research dependencies
├── requirements-notebook.txt         # notebook/kernel execution dependencies
├── pytest.ini
├── .gitignore
├── .dockerignore
├── .python-version                   # Python 3.12
├── Dockerfile
├── compose.yaml
├── README.md
└── FILES.sha256                       # delivery file checksums
```

`.venv/` and `runs/` appear locally after installation/execution; they are generated,
ignored by Git and excluded from the delivery. The two raw datasets are included
unchanged in the ZIP for convenience, but remain ignored by Git. Their hashes
match the approved release metadata. Their original notices remain in the CSVs.

Test transcripts use a `.log` extension and test discovery excludes generated
reports/data/notebook/run folders. This fixes a packaging issue where a saved
`test-results.txt` transcript could otherwise be collected as a doctest.

The small approved model is intentionally tracked together with its metadata and
manifest in this final layout so a repository checkout can serve the release.
Do not commit every training run or your virtual environment. The Docker context
excludes raw data, notebooks, tests, examples, reports and temporary runs.

Keeping the runtime modules at the root is deliberate. The serialized pipeline
imports custom transformers as `aps_pipeline.FeatureEngineer` and
`aps_pipeline.MissingnessPruner`; preserving that path avoids an unnecessary model
and import migration. A `src/` directory is not what makes a project production
grade. Reproducibility, correctness, clear responsibilities and verification matter.

## 3. What to keep, replace, archive or remove

Nothing was deleted from your original project. First keep an external backup
of the entire old project, including its Git history. Do the migration on a real
branch and review the diff. Only remove obsolete active files after the new app,
example and notebook work on your machine.

| Existing item | Action | Reason |
|---|---|---|
| `.git/` | **Keep in place. Never replace or delete.** | Preserves authentic commit history and remotes. |
| Old `app.py` | Replace with delivered root `app.py` | Removes fabricated fleet data, unsupported calibration and mismatched metrics. |
| Old `README.md` / outer `readme.md` | Replace active documentation with delivered README and this guide | Old paths/models/claims did not match the executable project. |
| Old `requirements.txt` | Replace with all three delivered requirement files | Exact runtime, research/test and notebook environments have different roles. |
| Old `.gitignore` containing `*` | Replace with delivered `.gitignore` | The wildcard ignored every untracked file. |
| `scania_rf_best.pkl` | Archive outside the active repo, then remove active copy | Incomplete old raw-input workflow; the new app uses `artifacts/aps_pipeline.joblib`. |
| `scania_scaler.pkl` | Archive, then remove active copy | Scaler is now inside the complete serialized pipeline. |
| `scania_pca.pkl` | Archive, then remove active copy | RF does not consume PCA; PCA remains a research branch. |
| `scania_aps_project.ipynb` | Archive as a historical notebook outside the active repo | It has the old preprocessing/validation order and incomplete artifact export. |
| Old proposed two-student notebook names | Do not create empty placeholder files | That README architecture was not the supplied implemented project. |
| `eda_plots/` | Archive as legacy evidence, then remove from the active layout | Current research figures are generated into `experiment-results/`; cohorts/transforms differ. |
| `scania_logo.svg` | Archive/remove from active layout | New UI does not use it or imply an official Scania product. |
| Existing `dataset/raw_data/*.csv` | Keep unchanged, in the delivered path | Same raw files are already included and checked by hash. Do not combine data versions. |
| `csac/` | Keep outside migration until verified; then remove the obsolete environment | Recreate a Python 3.12 `.venv/`; old 3.13 artifacts/environment are not this tested release. |
| `.idea/`, `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/` | Optional local cleanup; never commit | Regenerable IDE/cache files. |
| Old ZIPs and earlier manual-upgrade packages | Keep outside the active repository | Prevents multiple conflicting “final” implementations and repository bloat. |

Your earlier directory listing put `.git` in the parent
`202618052_and_202618047_ml_project/`, above the old application folder.
Confirm the actual root on your machine with `git rev-parse --show-toplevel`.
Copy the **contents** of the delivered folder into that Git root, preserving the
existing `.git`. Do not accidentally create another application folder one level
below it, or put `.github/workflows/` below the actual Git root.

Do not reinitialize Git or fabricate/backdate commits. Make honest incremental
commits for the migration, notebook, checks and documentation. I could not inspect
your original external Git history due to access restrictions.

If raw CSVs or the old environment were already tracked, `.gitignore` does not
untrack them. Inspect `git ls-files` first. Use `git rm --cached -- <exact-tracked-path>`
only for files you intend to stop tracking; `--cached` keeps the local file.
Avoid blanket delete/untrack commands.

## 4. Why the pipeline changed

| Area | Old behavior | Final behavior and reason |
|---|---|---|
| Ingestion | UCI CSV parsing in notebook | Shared `load_uci` keeps the 20-line skip and `na` mapping, validates labels and 170 named columns. |
| Split order | Preprocessing fitted before validation split | Raw records split 85/15 first; learned transformations are fitted on 51,000 fit rows. Prevents validation leakage. |
| Feature engineering | Row missingness and histogram summaries | Same design; all-missing histogram sums remain NaN using `min_count=1` rather than becoming false zeros. |
| Missingness pruning | Fitted on all 60,000 development rows | Fitted on fit rows only; stored transformer owns kept columns. |
| Imputation | Fitted median not exported | Median imputer is inside the saved pipeline. Same statistics are applied at inference. |
| Constant features | Selector not exported | Selector is inside the saved pipeline. Feature order cannot be reconstructed by guesswork. |
| Scaling | Separate scaler; blanket sanitization of anomalies | Fitted RobustScaler inside pipeline; invalid upload values are rejected. No invented replacement of bad readings with zero. |
| RF representation | RF actually bypassed PCA, despite broad PCA claims | RF uses scaled sensor/engineered space. PCA is described and tested as a separate research branch. |
| Threshold | Coarse validation sweep mixed with test cost plot | Exact unique validation scores, consistent ties and extreme policies; chosen threshold is frozen in release metadata. |
| Final refit | RF refitted on 60,000 after threshold selection | Preserve the exact 51,000-fit model/threshold pair whose behavior was validated. |
| Persistence | RF/scaler/PCA only | One complete raw-input pipeline, metadata, validation evidence and integrity manifest. |
| Dashboard probabilities | Arbitrary 1.8% floor and nonlinear score remapping | Display honest uncalibrated model scores. No invented Bayesian or calibration claim. |
| Test metrics | Hardcoded tables disagreeing with notebook | Read measured results generated for the actual frozen release. |

The production pipeline is:

```text
raw named sensor records
    → row-wise feature engineering
    → fit-owned missingness pruning
    → fit-owned median imputation
    → fit-owned constant-feature removal
    → fit-owned RobustScaler
    → fitted balanced-subsample Random Forest
    → unchanged model score
    → frozen validation-selected decision threshold
```

The source CSVs contain 59,000 negative and 1,000 positive training records.
The fit split has 50,150 negatives and 850 positives; validation has 8,850 and
150. The cost metric is `10 FP + 500 FN`. Choosing a cutoff empirically is
cost-sensitive threshold selection, not Bayesian optimization.

For a calibrated posterior with zero loss on correct predictions, comparing
`10(1-p)` with `500p` gives `p > 10/510`. The deployed scores have not been
calibrated to deployment posteriors, so we do not blindly use that theoretical
cutoff. Class weighting and resampling can affect calibration.

The old “every truck has at least 1.8% conditional risk” argument was unsupported:
population prevalence is not a lower bound on each conditional probability.
A classifier can emit 0 or 1 without establishing physical impossibility/certainty.
RF probabilities average leaf class probabilities, not necessarily hard tree votes.

## 5. What changed in the application and why

The app has four workspaces: **Fleet review**, **Policy analysis**, **Research
diagnostics**, and **Model record**.

- Real CSV upload replaces six hand-crafted scaled vectors. Input is always the
  original 170 named sensor columns, with optional unique `record_id`.
- Explicit validation/scoring produces a real inspection-review queue. Users can
  filter flags, inspect raw readings, clear results and export the visible queue.
- A model score stays the same when a policy threshold changes. The policy page
  calculates scenarios from saved validation labels/scores; it does not alter the
  approved model or invent exponential cost curves.
- The research page selects real generated model and cluster results. No named
  physical failure mechanisms are inferred from anonymized sensor names.
- The record page shows actual model metrics and provenance. It does not display
  manually entered trophy tables, fabricated routes/mileage, a 24-hour breakdown
  forecast, component-replacement instructions or highway clearance.
- Quiet native Streamlit styling replaces external Bootstrap/CDN injection,
  gradients, oversized promotional banners and excessive gauges/icons.
- Status labels are “Review for inspection” and “Below threshold.” UCI negatives
  mean non-APS-related failures, not universally healthy trucks.

The five example records are real training-file examples with generated IDs.
Their expected scores are provided for a smoke test, not a new evaluation claim.

## 6. What changed in research and why

`experiments.py` now actually trains five development comparisons on the same raw
split: balanced RF, balanced HGB, balanced log-PCA LR, SMOTE log-PCA LR and
undersampled log-PCA LR. Previously, resampled arrays were created without being
consumed by the benchmark models.

The PCA/distance branch uses a checked nonnegative `log1p` transformation followed
by robust scaling and 50 PCs. This investigates extreme-value dominance: the old
unlogged PCA's near-100% variance in 50 components was not proof that fault
information was retained. RobustScaler does not bound transformed outliers or
make PCA itself robust. Each experiment has independent transformer instances.

SMOTE uses minority/majority ratio 0.20 and undersampling ratio 0.5. On the fit
partition those imply 10,030/50,150 minority/majority rows for SMOTE (60,180 total),
and 850/1,700 for undersampling (2,550 total). The original full-60,000 resample
counts are therefore not the new training-fold counts. Resampled LR branches
do not also apply balanced class weights.

The script adds train/validation minority-F1 depth curves, both ROC and PR plots,
PCA variance, k=2/3/4 clustering scores and sizes, and a seeded Ward subset.
Clustering uses only 850 fit-partition APS failures, not all 1,000 labels.
Current log-PCA silhouette scores are approximately 0.2653, 0.2729 and 0.3030;
k=4 is best among these candidates, with sizes 154/293/83/320. These are
exploratory statistical groups. They do not prove four mechanical failure modes.

Validation costs were RF 4,510; HGB 6,100; balanced log-PCA LR 7,840; SMOTE LR
8,280; undersampled LR 7,740. These reuse one validation split and do not provide
unbiased model-search performance. No research result automatically replaces the
approved serving release. Repeated folds, stability analysis and external labels
remain useful further studies.

## 7. Exactly what to do with the notebook

Use **`notebooks/01_scania_aps_end_to_end.ipynb`**. Do not paste new scripts into
the old notebook or manually transplant its old fitted variables into the new app.
The old notebook is a historical archive; it is not the active source of truth.

The new notebook has 28 cells (14 executable code cells). It covers:

1. Repository/environment discovery and an explicit rebuild/inspection mode.
2. Real UCI ingestion, label counts and raw EDA.
3. A raw stratified split before learned preprocessing.
4. Training/export through `train.py` and inspection of the complete pipeline.
5. Manual cross-checks of fit-only pruning/medians and frozen validation scores.
6. Validation cost/threshold selection, then exact frozen test metric reproduction.
7. Actual supervised/resampling/PCA experiments through `experiments.py`.
8. Complexity, clustering and seeded Ward plots with defensible interpretations.
9. The same example upload/scoring functions used by the app.
10. Full integration verification through `verify_release.py` and a run summary.

`REBUILD_RELEASE=True` and `REBUILD_EXPERIMENTS=True` are the defaults. Run All
performs real training and research regeneration into a fresh `runs/<run-id>/`.
The approved `artifacts/` and `experiment-results/` folders are not overwritten.
Set both flags to False for a quicker walkthrough of the shipped outputs.
Even in inspection mode the notebook reproduces test scores and verifies them.

If you later change model/preprocessing code, change the shared modules first,
retrain into a fresh versioned release, rerun the notebook/tests, inspect the new
metrics, then explicitly promote an approved release. Do not hand-edit notebook
numbers, metadata thresholds or checksums to make old artifacts appear compatible.
Do not rename/move custom transformer modules while retaining the old pickle.

## 8. Installation, notebook environment and local run

Use **CPython 3.12**. Open a terminal at the actual repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name aps-project --display-name "APS Project (Python 3.12)"
```

In PyCharm or VS Code select `.venv` / the registered APS kernel, open the notebook
in `notebooks/`, and use Run All. The delivered notebook includes the outputs from
an actual full regeneration run; its execution evidence is in reports/.

Start the application:

```powershell
$env:APS_AUTH_MODE = 'local'
.\.venv\Scripts\python.exe launch.py
```

Open http://127.0.0.1:8501. Upload `examples/aps_sample_input.csv`; compare its
scores/actions with `examples/expected_sample_scores.csv`. No source edits are
needed. Local mode has no login and `launch.py` restricts it to loopback.
Do not expose that mode publicly or through a tunnel.

The application accepts plain UTF-8 CSVs with one header row, the 170 raw sensor
columns, and optional unique nonblank `record_id`. Blank or literal `na` denotes
missing data. The UCI metadata preamble and `class` column belong only in the
training/evaluation files, not in an app upload. The template button gives headers.

## 9. Tests and measured model results

Run the suite and real-data verification:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=reports/test-results.xml
.\.venv\Scripts\python.exe verify_release.py --test dataset/raw_data/aps_failure_test_set.csv --out reports/verification.json
```

Evidence files in `reports/` distinguish the following checks:

The final repository passed **50 automated tests**, including a rerun with pytest
9.0.3. Its notebook
completed all **14 executable cells** with both rebuild flags True in 168.38 seconds
and no error outputs. These are actual recorded runs, not syntax-only checks.

- The automated suite tests transformations, leakage boundaries, costs, tied
  thresholds, serialization, invalid uploads, formula-safe exports, corrupt model
  files, version mismatches, verified-email access, safe startup, HTTP health,
  UI state navigation/filtering/details, and shipped example-to-model parity.
- AppTest supplies a mocked file-uploader byte stream; parsing and model scoring
  are real. Native browser upload/download and visual styling were not automated.
- Real-data integration reproduces the entire 16,000-row test set, checks raw CSV
  scoring/export parity, runs four concurrent calls and processes 10,000 records.
- The new notebook is executed with both rebuild flags True; training and research
  are genuinely run, and every code cell must complete with no error output.
- The 44 runtime pins were checked against PyPI per-version advisory data and none
  were listed. This is not an OS/image scan, penetration test or vulnerability-free
  guarantee. The additional 39 development/notebook pins were also checked after
  updating pytest; none had listed advisories. The separate JSON reports retain
  the package-by-package results.

| Official test result | Value |
|---|---:|
| Threshold selected from validation | 0.11 |
| TN / FP / FN / TP | 15,393 / 232 / 28 / 347 |
| ROC-AUC | 0.99334 |
| Average precision | 0.88180 |
| Recall / precision | 0.92533 / 0.59931 |
| Minority F1 | 0.72746 |
| Selected-policy benchmark cost | 16,320 |
| Same model at threshold 0.50 | 76,640 |

The selected-policy arithmetic is `232×10 + 28×500 = 16,320`.
The same-model comparison is a 78.7% reduction versus 0.50, not measured annual
fleet savings. The new cost is **higher** than the old notebook's stored 11,260.
That legacy result used a different preprocessing/selection procedure; we do not
claim this release beats it or tune on the test to recover a better-looking number.

The ordinary paired 1,000-resample bootstrap gave a test-cost 95% percentile
interval [11,680, 21,800] and saving-versus-0.50 interval [49,369.25, 71,771].
Those intervals condition on this benchmark and fixed model, not future domain
shift or the uncertainty from retraining/model search.

On the measured Windows execution environment, CSV parsing plus scoring took
roughly 42 ms for one row, 91 ms for 1,000 rows and 618 ms for 10,000 rows at the
median of three runs. Network/login/rendering and hosting load are excluded.
Fresh verification writes fresh latency evidence; these are not SLA guarantees.

There are known nonfatal research-environment messages: sklearn/SciPy reports an
unused `iprint` optimization option in LR fits, and Windows notebook execution may
emit a ZMQ selector-thread warning. Fits completed; convergence warnings are not
globally suppressed. Notebook subprocess stderr logs are saved in its runs/ folder.

## 10. Production safeguards added

- Strict named-schema input checks: duplicate/missing/extra headers, malformed row
  widths, invalid UTF-8/numbers, infinities, extreme values and entirely missing
  records are rejected. Reordered columns are aligned by name.
- IDs preserve leading zeros; blanks, duplicates, excessive length and control
  characters are rejected. Exported text fields are protected from spreadsheet
  formula prefixes.
- Uploads are limited to 10 MiB / 10,000 records. Each prediction uses one RF
  worker; the process admits at most two concurrent scoring calls.
- Session results are isolated and invalidated when the upload/release changes.
  The app does not write uploads to disk. The clear action removes scored results;
  users must remove the upload to clear uploader contents.
- Hosted mode defaults to OIDC plus an explicit verified-email allowlist and
  fails closed when unconfigured. Local demo mode is explicitly separate.
- Release loading checks file hashes, Python/scientific package versions,
  transformer-source hash, schema and class mapping. Only trusted deployment
  artifacts are loaded; users cannot upload pickles.
- Training stages a complete release, verifies reload parity, then atomically
  publishes it. Existing output directories are not overwritten.
- Metadata records seed, parameters, dataset/split hashes, sizes, creation time,
  threshold and measured results. A checksum detects corruption, not replacement
  by an attacker who can also replace the manifest; protect release write access.
- Logs record model ID, row count and duration; uploaded filenames/IDs/readings
  are not deliberately logged. Host logs/backups and session retention still need
  your deployment policy.

## 11. Hosted setup, rollout and rollback

For hosted use, register an OIDC client with your organization's identity provider,
set the redirect URI to `https://YOUR-HOST/oauth2callback`, and copy
`.streamlit/secrets.toml.example` to a real `.streamlit/secrets.toml` **on the host**.
Replace placeholders with real values and a long cryptographically random cookie
secret. Never commit real secrets. List authorized accounts under
`[access].allowed_emails`. This release requires `email_verified = true`; a provider
that omits that claim is denied until its claim mapping is deliberately adapted.

Unset the local-mode variable or set `APS_AUTH_MODE=oidc`. `launch.py` checks
configuration and release validity before serving. Managed Streamlit hosting may
use `app.py` as its entrypoint and its own secrets UI; the app also checks access
and release validity. Configure an HTTPS reverse proxy with WebSocket support,
request/upload limits, concurrency controls and an appropriate idle-session policy.

For Docker:

```bash
docker compose up --build -d
docker compose logs -f aps
```

The templates use Python 3.12, a non-root user, a read-only filesystem, dropped
capabilities, CPU/memory/process limits, read-only mounted secrets and a liveness
probe. Docker and Linux hosting were **not executed** in this environment; verify
the image, identity-provider callback, permissions and logs on the actual host.
The checked-in GitHub Actions workflow is a definition, not proof a remote CI run
has passed. Run it in your repository and retain its result.

For a new model, use a fresh destination:

```powershell
.\.venv\Scripts\python.exe train.py --train dataset/raw_data/aps_failure_training_set.csv --test dataset/raw_data/aps_failure_test_set.csv --out runs/aps-v2/artifacts
```

Review its development evidence and verification before promotion. Select an
approved immutable directory with `APS_ARTIFACT_DIR`, stop/restart the app, and
verify the expected model/threshold. Roll back by selecting the prior approved
directory and restarting. Do not edit a live model, threshold or manifest in place.
Notebook Run All never automatically promotes a research model.

## 12. Remaining external checks and honest scope

The notebook, pipeline, tests and app-server checks can be completed locally.
These items require your real deployment/data and are not claimed as completed:

1. Successful authorized OIDC login/logout/callback and rejection of an actual
   unauthorized account, including provider-specific claims/session expiry.
2. Docker/Linux build/run, actual GitHub CI execution and the final host's TLS,
   WebSocket, proxy, resource and secret settings.
3. Visual desktop/mobile rendering, a native browser upload and an actual download.
   The connected browser could not access the isolated loopback server. AppTest
   states and HTTP checks do not prove pixel-level appearance.
4. Prospective fleet validation, agreement on allowable missed failures/inspection
   load, and monitoring against delayed real outcome labels. The official benchmark
   has already been observed during the broader project; obtain new external data
   for a fresh deployment-generalization claim.

No live telemetry connector, durable work-order database, tenant billing,
remaining-useful-life model, automatic repair action or operational certification
is included. Use actual timestamps/truck IDs and grouped/time-separated validation
when available. Monitor availability, latency, errors, missingness, score shifts
and labeled cost/recall; drift alone is not proof of performance loss.

**Status:** one tested, coherent deployment candidate with a matching executable
notebook, ready for staging/host acceptance. It is not honest to call every possible
deployment and commercial condition tested without the real target environment.

## 13. Final migration checklist

- Keep the existing `.git` and an external backup of the old project.
- Extract this final ZIP and use its contents as the active repository layout.
- Keep exactly one active notebook under `notebooks/`; archive the old one.
- Install Python 3.12 requirements; do not reuse the old 3.13 virtual environment.
- Run the tests, sample upload and notebook from the delivered layout.
- Compare model-record metrics with the evidence in reports/.
- Keep raw datasets locally but out of Git; keep the approved small model and its
  matching provenance together.
- Configure real hosted access and complete the external checks before going live.
- Commit real reviewed changes; do not reconstruct fictitious development history.

Dataset/task: https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks
Leakage guidance: https://scikit-learn.org/stable/common_pitfalls.html
Persistence: https://scikit-learn.org/stable/model_persistence.html
App testing: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest
Hosted login: https://docs.streamlit.io/develop/api-reference/user/st.login
