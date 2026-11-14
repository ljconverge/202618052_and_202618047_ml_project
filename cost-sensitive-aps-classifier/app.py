"""APS review workspace. Hosted mode requires OIDC plus a user allowlist."""
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time
import pandas as pd
import streamlit as st
from metrics import cost_row, threshold_curve
from service import (export_csv, is_allowed_user, load_release, parse_upload,
                     release_fingerprint, score_batch)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
st.set_page_config(page_title="APS | Fleet review", layout="wide")
ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("APS_ARTIFACT_DIR", str(ROOT / "artifacts")))


def require_access():
    mode = os.environ.get("APS_AUTH_MODE", "oidc")
    if mode == "local":
        st.caption("Local demonstration mode · bind the server to 127.0.0.1")
        return
    if mode != "oidc":
        st.error("Access configuration is invalid.")
        st.stop()
    try:
        allowed = st.secrets.get("access", {}).get("allowed_emails", [])
    except Exception:
        allowed = []
    if not allowed:
        st.error("Access has not been configured. Contact the administrator.")
        st.stop()
    if not st.user.is_logged_in:
        st.title("APS fleet review")
        st.write("Sign in with your organization account.")
        if st.button("Sign in", type="primary"):
            st.login()
        st.stop()
    if not is_allowed_user(dict(st.user), allowed):
        st.error("This account does not have access.")
        if st.button("Sign out"):
            st.logout()
        st.stop()
    with st.sidebar:
        if st.button("Sign out"):
            st.logout()


require_access()


@st.cache_resource
def cached_release(path, fingerprint):
    return (*load_release(path), threading.BoundedSemaphore(2))


try:
    model, meta, evidence, capacity = cached_release(str(ARTIFACTS), release_fingerprint(ARTIFACTS))
except Exception:
    logging.exception("release_load_failed")
    st.error("Model release unavailable. Contact the administrator.")
    st.stop()

with st.sidebar:
    st.title("APS")
    st.caption("Fleet inspection support")
    page = st.radio("Workspace", ["Fleet review", "Policy analysis", "Research diagnostics", "Model record"], key="page")
    st.divider()
    st.caption(f"Release: {meta['model_id']}")
    st.caption(f"Decision threshold: {meta['threshold']:.3f}")
    st.caption("Independent Scania dataset study")

if page == "Fleet review":
    st.title("Fleet review")
    st.caption("Upload records, review inspection flags, and export the queue.")
    st.info("A score indicates evidence for APS-related failure. A low score does not certify vehicle health.")
    st.download_button("Download CSV header template",
                       pd.DataFrame(columns=meta["raw_features"]).to_csv(index=False),
                       "aps_input_template.csv", "text/csv")
    uploaded = st.file_uploader("Sensor CSV", type=["csv"], key="sensor_upload")
    st.caption("170 raw sensor columns; optional unique record_id. Blank or 'na' means missing. Maximum 10 MB / 10,000 records. Uploaded records are not written to disk by this app.")
    if st.button("Clear scored batch"):
        st.session_state.pop("scored_batch", None)
        st.session_state.pop("detail_record", None)
        st.success("Scored results cleared. Remove the file from the uploader to clear its contents.")
    if uploaded is None:
        st.session_state.pop("scored_batch", None)
        st.stop()
    payload = uploaded.getvalue()
    batch_key = (hashlib.sha256(payload).hexdigest(), release_fingerprint(ARTIFACTS))
    existing = st.session_state.get("scored_batch")
    if existing and existing["key"] != batch_key:
        st.session_state.pop("scored_batch", None)
        st.session_state.pop("detail_record", None)
    with st.form("score_form"):
        submit = st.form_submit_button("Validate and score batch", type="primary")
    if submit:
        st.session_state.pop("scored_batch", None)
        if not capacity.acquire(blocking=False):
            st.warning("Scoring capacity is busy. Try again shortly.")
        else:
            start = time.perf_counter()
            try:
                with st.spinner("Validating and scoring records…"):
                    raw, ids = parse_upload(payload, meta["raw_features"])
                    result = score_batch(model, meta, raw, ids)
                st.session_state["scored_batch"] = dict(key=batch_key, result=result, raw=raw)
                logging.info("batch_scored rows=%d elapsed_ms=%.1f model=%s", len(raw),
                             (time.perf_counter() - start) * 1000, meta["model_id"])
            except ValueError as exc:
                st.error(str(exc))
            except Exception:
                logging.exception("batch_scoring_failed")
                st.error("Scoring failed. Contact the administrator.")
            finally:
                capacity.release()
    batch = st.session_state.get("scored_batch")
    if not batch:
        st.stop()
    result, raw = batch["result"], batch["raw"]
    a, b, c = st.columns(3)
    a.metric("Records", len(result))
    b.metric("Flagged for review", int((result.model_score >= meta["threshold"]).sum()))
    c.metric("Median missing sensors", f"{result.missing_fraction.median():.1%}")
    view = st.radio("Queue", ["All", "Flagged"], horizontal=True, key="queue_filter")
    shown = result if view == "All" else result[result.action == "Review for inspection"]
    shown = shown.sort_values("model_score", ascending=False)
    st.dataframe(shown, hide_index=True, width="stretch",
                 column_config={"model_score": st.column_config.NumberColumn("Model score", format="%.3f"),
                                "missing_fraction": st.column_config.NumberColumn("Missing fraction", format="%.3f")})
    st.download_button("Export current queue", export_csv(shown), "aps_review_queue.csv", "text/csv")
    st.subheader("Record details")
    selected = st.selectbox("Record", result.record_id.tolist(), key="detail_record")
    idx = result.index[result.record_id == selected][0]
    st.caption("Raw anonymized readings; sensor names do not establish a mechanical diagnosis.")
    st.dataframe(raw.iloc[idx].rename("reading").to_frame(), width="stretch")

elif page == "Policy analysis":
    st.title("Policy analysis")
    st.caption("Validation evidence only. Sandbox changes do not change the deployed policy.")
    t = st.slider("Scenario threshold", 0., 1., min(1., float(meta["threshold"])), .001)
    current = cost_row(evidence.label, evidence.score, t)
    a, b, c = st.columns(3)
    a.metric("False positives", current["fp"])
    b.metric("False negatives", current["fn"])
    c.metric("Validation cost units", f"{current['cost']:,}")
    curve = threshold_curve(evidence.label, evidence.score)
    st.line_chart(curve.set_index("threshold")[["cost"]])
    st.caption("Loss = 10 × false positives + 500 × false negatives. These benchmark costs are not measured annual savings.")
    st.caption(f"Frozen release threshold: {meta['threshold']:.3f}. Scores are uncalibrated.")

elif page == "Research diagnostics":
    st.title("Research diagnostics")
    st.caption("Exploratory development results. These experiments do not change the deployed model.")
    directory = ROOT / "experiment-results"
    if not (directory / "validation_benchmarks.csv").exists():
        st.info("Development experiment results are not included in this deployment.")
        st.stop()
    benchmarks = pd.read_csv(directory / "validation_benchmarks.csv")
    selected_model = st.selectbox("Development model", benchmarks.model.tolist())
    st.dataframe(benchmarks[benchmarks.model == selected_model], hide_index=True, width="stretch")
    st.bar_chart(benchmarks.set_index("model")[["cost"]])
    clusters = json.loads((directory / "clusters.json").read_text())
    k = st.selectbox("Exploratory cluster count", [item["k"] for item in clusters])
    selected_cluster = next(item for item in clusters if item["k"] == k)
    a, b = st.columns(2)
    a.metric("Silhouette", f"{selected_cluster['silhouette']:.3f}")
    b.metric("Davies–Bouldin", f"{selected_cluster['davies_bouldin']:.3f}")
    st.bar_chart(pd.DataFrame({"size": selected_cluster["sizes"]}).rename_axis("cluster"))
    st.caption("Clusters group anonymous observations; they do not establish physical failure mechanisms. PCA uses a separate log1p branch.")
    st.subheader("Tree complexity and dimensionality")
    st.line_chart(pd.read_csv(directory / "complexity.csv").set_index("depth"))
    st.image(str(directory / "pca.png"), width="stretch")
    with st.expander("Validation ROC and precision–recall curves"):
        st.image(str(directory / "roc_pr.png"), width="stretch")

else:
    st.title("Model record")
    st.caption(f"Built {meta['created_utc']} · release {meta['model_id']}")
    st.dataframe(pd.DataFrame(meta["metrics"]).T, width="stretch")
    st.write("All learned preprocessing was fitted on the 85% fit partition. The threshold was selected on a separate stratified 15% validation partition. The same fitted model was then evaluated on the official test data.")
    st.write("Scope: APS-related versus non-APS-related failure classification. The benchmark does not establish breakdown timing, remaining useful life, replacement instructions, or dispatch safety.")
    with st.expander("Release provenance"):
        st.json({k: v for k, v in meta.items() if k != "metrics"})
