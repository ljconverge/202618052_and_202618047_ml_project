import os
from pathlib import Path
import subprocess
import sys
import socket
import time
import http.client
import pytest

LAUNCH = str(Path(__file__).resolve().parents[1] / "launch.py")


@pytest.mark.parametrize("mode,address,message", [
    ("invalid", "127.0.0.1", "must be local or oidc"),
    ("local", "0.0.0.0", "must bind to 127.0.0.1"),
    ("oidc", "127.0.0.1", "Configure valid OIDC secrets"),
])
def test_startup_rejects_unsafe_configuration(mode, address, message):
    env = dict(os.environ, APS_AUTH_MODE=mode, APS_BIND_ADDRESS=address)
    result = subprocess.run([sys.executable, LAUNCH], env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode != 0
    assert message in result.stderr


def test_local_launcher_serves_health_and_html(release, tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, APS_AUTH_MODE="local", APS_BIND_ADDRESS="127.0.0.1",
               APS_PORT=str(port), APS_ARTIFACT_DIR=str(release))
    logfile = tmp_path / "server.log"
    with logfile.open("w") as log:
        child = subprocess.Popen([sys.executable, LAUNCH], env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                    conn.request("GET", "/_stcore/health")
                    response = conn.getresponse()
                    assert response.status == 200 and response.read() == b"ok"
                    conn.close()
                    break
                except (OSError, http.client.HTTPException):
                    assert child.poll() is None, logfile.read_text()
                    if time.monotonic() > deadline: raise
                    time.sleep(.2)
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/")
            response = conn.getresponse()
            assert response.status == 200 and b"streamlit" in response.read().lower()
            conn.close()
        finally:
            child.terminate()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill()
