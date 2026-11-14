from io import BytesIO
from pathlib import Path
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def app(monkeypatch, release):
    monkeypatch.setenv("APS_AUTH_MODE", "local")
    monkeypatch.setenv("APS_ARTIFACT_DIR", str(release))
    return AppTest.from_file(APP, default_timeout=30).run()


def test_pages_and_policy_reactivity(monkeypatch, release):
    at = app(monkeypatch, release)
    assert not at.exception
    assert any(item.value == "Fleet review" for item in at.title)
    at.radio(key="page").set_value("Policy analysis").run()
    assert not at.exception
    at.slider[0].set_value(0.).run()
    assert at.metric[0].value == "1"  # all records flagged: one FP
    at.slider[0].set_value(1.).run()
    assert at.metric[1].value == "1"  # no records flagged: one FN
    at.radio(key="page").set_value("Model record").run()
    assert not at.exception
    assert len(at.dataframe) == 1


def test_research_diagnostics(monkeypatch, release):
    at = app(monkeypatch, release)
    at.radio(key="page").set_value("Research diagnostics").run()
    assert not at.exception
    if at.info: return  # Optional research artifacts are not required for serving.
    at.selectbox[0].set_value("HGB balanced").run()
    assert at.dataframe[0].value.iloc[0]["model"] == "HGB balanced"
    at.selectbox[1].set_value(4).run()
    assert not at.exception
    assert float(at.metric[0].value) > 0


def test_auth_default_fails_closed(monkeypatch, release):
    monkeypatch.delenv("APS_AUTH_MODE", raising=False)
    monkeypatch.setenv("APS_ARTIFACT_DIR", str(release))
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert "Access has not been configured" in at.error[0].value
    assert len(at.radio) == 0


def test_upload_score_filter_detail_export_clear(monkeypatch, release, raw):
    # AppTest cannot directly manipulate a file_uploader in this Streamlit
    # version. Supply its byte stream; all parsing and model scoring run normally.
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: BytesIO(raw.to_csv(index=False).encode()))
    at = app(monkeypatch, release)
    at.button(key="FormSubmitter:score_form-Validate and score batch").click().run()
    assert not at.exception and not at.error
    assert at.metric[0].value == "30"
    assert len(at.dataframe[0].value) == 30
    at.radio(key="queue_filter").set_value("Flagged").run()
    assert not at.exception
    assert set(at.dataframe[0].value.action) <= {"Review for inspection"}
    at.selectbox(key="detail_record").set_value("row-2").run()
    assert not at.exception
    [b for b in at.button if b.label == "Clear scored batch"][0].click().run()
    assert not at.exception
    assert len(at.metric) == 0


def test_invalid_upload_does_not_render_old_results(monkeypatch, release):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: BytesIO(b"bad\nvalue"))
    at = app(monkeypatch, release)
    at.button(key="FormSubmitter:score_form-Validate and score batch").click().run()
    assert not at.exception
    assert len(at.error) == 1 and len(at.metric) == 0
