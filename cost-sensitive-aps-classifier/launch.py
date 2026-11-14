"""Fail before serving if the release or hosted access configuration is invalid."""
import os
from pathlib import Path
import sys
import tomllib
from service import load_release

root = Path(__file__).resolve().parent
os.chdir(root)
mode = os.environ.get("APS_AUTH_MODE", "oidc")
address = os.environ.get("APS_BIND_ADDRESS", "127.0.0.1" if mode == "local" else "0.0.0.0")
port = int(os.environ.get("APS_PORT", "8501"))
if not 1 <= port <= 65535:
    raise SystemExit("APS_PORT must be between 1 and 65535")
if mode not in {"local", "oidc"}:
    raise SystemExit("APS_AUTH_MODE must be local or oidc")
if mode == "local" and address != "127.0.0.1":
    raise SystemExit("Local demonstration mode must bind to 127.0.0.1")
if mode == "oidc":
    secrets_path = root / ".streamlit" / "secrets.toml"
    try:
        secrets = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
        auth = secrets["auth"]
        if not secrets["access"]["allowed_emails"]:
            raise ValueError("No authorized users")
        if not all(auth.get(k) for k in ("redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url")):
            raise ValueError("Missing OIDC settings")
        if not auth["redirect_uri"].startswith("https://") or not auth["server_metadata_url"].startswith("https://"):
            raise ValueError("Hosted authentication requires HTTPS")
        if len(auth["cookie_secret"]) < 32:
            raise ValueError("Cookie secret must contain at least 32 characters")
        if any("YOUR-" in str(v) or "GENERATE-" in str(v) for v in auth.values()):
            raise ValueError("Replace example credentials")
    except Exception as exc:
        raise SystemExit("Configure valid OIDC secrets and allowed_emails before hosted startup.") from exc
load_release(os.environ.get("APS_ARTIFACT_DIR", str(root / "artifacts")))
from streamlit.web import cli
sys.argv = ["streamlit", "run", str(root / "app.py"), f"--server.address={address}",
            f"--server.port={port}", "--server.headless=true"]
raise SystemExit(cli.main())
