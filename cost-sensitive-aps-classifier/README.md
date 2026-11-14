# Cost-sensitive APS fault classification

DS605 project by **Lakshita Pagaria (202618052)** and **Ameya Trivedi (202618047)**.
Independent study of the UCI Scania benchmark; no Scania affiliation is implied.

The application accepts raw anonymized sensor records, produces an inspection
review queue and evaluates validation-based decision policies. It distinguishes
APS-related from non-APS-related failures; it does not certify vehicle health or
predict a validated time-to-breakdown.

**Read [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) first.** That single guide
contains the repository structure, migration/delete/keep instructions, every
change and its reason, notebook instructions, measured results, test evidence,
deployment setup and the remaining host-specific checks.

## Run locally

Use Python **3.12**. Run commands from this repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:APS_AUTH_MODE = 'local'
.\.venv\Scripts\python.exe launch.py
```

Open http://127.0.0.1:8501 and upload `examples/aps_sample_input.csv` to try the
real scoring path. Those five example records come from training data; they are
not unseen validation or live telemetry. Local mode is loopback-only.

## Notebook and tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name aps-project --display-name "APS Project (Python 3.12)"
.\.venv\Scripts\python.exe -m pytest -q --junitxml=reports/test-results.xml
```

Open `notebooks/01_scania_aps_end_to_end.ipynb` in PyCharm, VS Code or Jupyter and
select that environment/kernel. Run All rebuilds the shared training pipeline
and research experiments into a fresh ignored `runs/` folder; the approved
`artifacts/` release is preserved. Both raw CSVs are included in this delivery.

The small approved model and its manifest are intentionally versioned together
in this repository layout so a source checkout can serve the release. Raw CSVs,
virtual environments, secrets and temporary training runs are ignored by Git.
Docker excludes raw data and notebooks from the serving image.

## Measured official-test result

| Threshold | ROC-AUC | Average precision | Recall | FP | FN | Cost |
|---:|---:|---:|---:|---:|---:|---:|
| 0.11 | 0.99334 | 0.88180 | 0.92533 | 232 | 28 | 16,320 |

Cost is `10 × FP + 500 × FN`. At 0.50, the same model costs 76,640. The original
notebook's reported 11,260 is lower; no improvement over that legacy number is
claimed. See the guide for methodology, uncertainty and limitations.

Hosted mode requires your real OIDC credentials, a verified-email allowlist and
HTTPS/proxy configuration. Live login, Docker/Linux deployment and visual mobile
acceptance still require checks on your target host. The tested code is a
deployment candidate, not a guarantee of operational fleet safety.
