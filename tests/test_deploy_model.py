"""deploy_model.sh tests. curl, docker, flock and crontab are replaced by stubs."""

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path("scripts/deploy_model.sh").resolve()
STUBS = {
    "curl": """#!/usr/bin/env bash
echo "curl $*" >> "$STUB_DIR/calls"
case "$*" in
  *registered-models/alias*) cat "$STUB_DIR/champion.json" ;;
  *get-artifact*)
    [ -f "$STUB_DIR/download_fails" ] && exit 22
    while [ $# -gt 0 ]; do
      [ "$1" = -o ] && echo "model-$(cat "$STUB_DIR/version")" > "$2"
      shift
    done ;;
  *health*)
    version=$(grep '^MODEL_VERSION=' .env | cut -d= -f2)
    grep -qx "$version" "$STUB_DIR/bad_versions" 2>/dev/null && exit 7
    echo "{\\"model\\": {\\"version\\": \\"$version\\"}}" ;;
esac
""",
    "docker": """#!/usr/bin/env bash
echo "docker $*" >> "$STUB_DIR/calls"
""",
    "flock": """#!/usr/bin/env bash
exit 0
""",
    "crontab": """#!/usr/bin/env bash
if [ "${1:-}" = "-l" ]; then
  cat "$STUB_DIR/crontab" 2>/dev/null
else
  cat > "$STUB_DIR/crontab.new"
  mv "$STUB_DIR/crontab.new" "$STUB_DIR/crontab"
fi
""",
}


class Deployment:
    def __init__(self, root):
        self.stubs = root / "stubs"
        self.home = root / "flight-delay"
        self.models = root / "models"
        for path in (self.stubs, self.home, self.models):
            path.mkdir()
        self.bin = root / "bin"
        self.bin.mkdir()
        for name, content in STUBS.items():
            path = self.bin / name
            path.write_text(content)
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
        shutil.copy(SCRIPT, self.home / "deploy_model.sh")
        (self.home / "compose.vm2.yml").write_text("services: {}\n")
        self.env_file = self.home / ".env"
        self.env_file.write_text("IMAGE_TAG=abc\nMLFLOW_URL=http://vm1:5000\n")
        self.model = self.models / "model.joblib"
        self.champion(3)

    def champion(self, version):
        (self.stubs / "version").write_text(str(version))
        (self.stubs / "champion.json").write_text(
            json.dumps(
                {"model_version": {"version": str(version), "run_id": f"run{version}"}}
            )
        )

    def run(self, *args, check=True):
        environment = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "STUB_DIR": str(self.stubs),
            "MODEL_DIR": str(self.models),
            "HEALTH_RETRIES": "1",
            "HEALTH_DELAY": "0",
        }
        return subprocess.run(
            ["bash", str(self.home / "deploy_model.sh"), *args],
            env=environment,
            capture_output=True,
            text=True,
            check=check,
        )

    @property
    def calls(self):
        path = self.stubs / "calls"
        return path.read_text().splitlines() if path.exists() else []

    def clear_calls(self):
        (self.stubs / "calls").write_text("")

    def docker_calls(self):
        return [call for call in self.calls if call.startswith("docker")]

    def installed(self, name="VERSION"):
        return (self.models / name).read_text().strip()

    def env(self):
        return dict(line.split("=", 1) for line in self.env_file.read_text().split())


@pytest.fixture
def deployment(tmp_path):
    return Deployment(tmp_path)


def test_first_run_installs_the_champion(deployment):
    deployment.run()

    assert deployment.model.read_text().strip() == "model-3"
    assert deployment.installed() == "3"
    assert deployment.env()["MODEL_VERSION"] == "3"
    assert deployment.env()["IMAGE_TAG"] == "abc"
    assert stat.S_IMODE(deployment.model.stat().st_mode) == 0o644
    assert any("get-artifact" in c and "run_uuid=run3" in c for c in deployment.calls)
    assert deployment.docker_calls() == [
        "docker compose -f compose.vm2.yml up -d --force-recreate"
    ]


def test_same_version_does_nothing(deployment):
    deployment.run()
    deployment.clear_calls()

    deployment.run()

    assert deployment.docker_calls() == []
    assert not any("get-artifact" in call for call in deployment.calls)


def test_new_champion_replaces_the_model(deployment):
    deployment.run()
    deployment.champion(4)

    deployment.run()

    assert deployment.model.read_text().strip() == "model-4"
    assert deployment.installed() == "4"
    assert deployment.env()["MODEL_VERSION"] == "4"


def test_unhealthy_model_is_rolled_back_and_not_retried(deployment):
    deployment.run()
    deployment.champion(4)
    (deployment.stubs / "bad_versions").write_text("4\n")

    result = deployment.run(check=False)

    assert result.returncode == 1
    assert deployment.model.read_text().strip() == "model-3"
    assert deployment.installed() == "3"
    assert deployment.installed("FAILED_VERSION") == "4"
    assert deployment.env()["MODEL_VERSION"] == "3"
    assert len(deployment.docker_calls()) == 3

    deployment.clear_calls()
    assert deployment.run().returncode == 0
    assert deployment.docker_calls() == []


def test_a_later_champion_is_deployed_after_a_failure(deployment):
    deployment.run()
    deployment.champion(4)
    (deployment.stubs / "bad_versions").write_text("4\n")
    deployment.run(check=False)
    deployment.champion(5)

    deployment.run()

    assert deployment.installed() == "5"
    assert not (deployment.models / "FAILED_VERSION").exists()


def test_failed_download_leaves_the_model_untouched(deployment):
    deployment.run()
    deployment.champion(4)
    (deployment.stubs / "download_fails").touch()
    deployment.clear_calls()

    result = deployment.run(check=False)

    assert result.returncode != 0
    assert deployment.model.read_text().strip() == "model-3"
    assert deployment.installed() == "3"
    assert deployment.docker_calls() == []


def test_missing_settings_stop_the_deployment(deployment):
    deployment.env_file.write_text("IMAGE_TAG=abc\n")

    result = deployment.run(check=False)

    assert result.returncode != 0
    assert "MLFLOW_URL" in result.stderr
    assert not deployment.model.exists()


def test_cron_runs_every_five_minutes_and_is_installed_once(deployment):
    deployment.run("--install-cron")
    deployment.run("--install-cron")

    lines = (deployment.stubs / "crontab").read_text().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("*/5 * * * * bash ")
    assert lines[0].endswith("deploy_model.log 2>&1")


def test_cron_keeps_other_entries(deployment):
    (deployment.stubs / "crontab").write_text("0 1 * * * /usr/bin/backup\n")

    deployment.run("--install-cron")

    lines = (deployment.stubs / "crontab").read_text().splitlines()
    assert lines[0] == "0 1 * * * /usr/bin/backup"
    assert len(lines) == 2
