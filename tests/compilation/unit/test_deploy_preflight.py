import subprocess
from pathlib import Path

import pytest

from sima_lmm.host import deploy_lmm, deploy_lora, remote_destination
from sima_lmm.host.remote_destination import RemoteDestination, parse_remote_destination


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


@pytest.mark.parametrize(
    ("dst", "expected"),
    [
        ("models/llama", None),
        ("/media/nvme/llima/llama", None),
        ("./host:dir", None),
        ("dir/with:colon", None),
        (":leading-colon", None),
        ("host::module/path", None),
        ("rsync://host/module/path", None),
        ("devkit:/media/nvme/llima", RemoteDestination("devkit", "/media/nvme/llima")),
        ("sima@192.168.1.20:/media/nvme", RemoteDestination("sima@192.168.1.20", "/media/nvme")),
        ("sima@devkit:models", RemoteDestination("sima@devkit", "models")),
        ("sima@devkit:", RemoteDestination("sima@devkit", "")),
        ("[::1]::module", None),
        ("sima@[::1]", None),
    ],
)
def test_parse_remote_destination(dst: str, expected: RemoteDestination | None) -> None:
    assert parse_remote_destination(dst) == expected


@pytest.mark.parametrize(
    ("path", "child"),
    [("/media/nvme/", "/media/nvme/model"), ("/", "/model"), ("", "model"), ("rel", "rel/model")],
)
def test_remote_child_path(path: str, child: str) -> None:
    assert RemoteDestination("devkit", path).child_path("model") == child


class FakeRun:
    """Records subprocess.run calls made by the remote session."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.calls = []
        self.result = (returncode, stdout, stderr)

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        returncode, stdout, stderr = self.result
        if "-O" in cmd:
            return subprocess.CompletedProcess(cmd, 0)
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    @property
    def preflight_calls(self):
        return [cmd for cmd in self.calls if "-O" not in cmd]

    @property
    def exit_calls(self):
        return [cmd for cmd in self.calls if "-O" in cmd]


def _write_sima_files(src: Path) -> Path:
    sima_dir = src / "sima_files"
    (sima_dir / "devkit").mkdir(parents=True)
    (sima_dir / "mpk").mkdir()
    (sima_dir / "devkit" / "vlm_config.json").write_text("{}")
    return sima_dir


@pytest.fixture
def fake_run(monkeypatch) -> FakeRun:
    monkeypatch.delenv("RSYNC_RSH", raising=False)
    fake = FakeRun()
    monkeypatch.setattr(remote_destination.subprocess, "run", fake)
    return fake


@pytest.fixture
def deploy_calls(monkeypatch) -> list:
    calls = []
    monkeypatch.setattr(
        deploy_lmm,
        "_deploy_sima_files",
        lambda src, dst, rsh=None: calls.append((src, dst, rsh)),
    )
    return calls


def test_preflight_runs_before_extraction_and_rsync_shares_connection(
    tmp_path: Path, fake_run: FakeRun, deploy_calls: list
) -> None:
    sima_dir = _write_sima_files(tmp_path / "compiled")

    deploy_lmm.deploy(tmp_path / "compiled", "sima@devkit:/media/nvme/llama")

    preflight = fake_run.preflight_calls
    assert len(preflight) == 1
    assert preflight[0][0] == "ssh"
    assert "ControlMaster=auto" in preflight[0]
    assert "sima@devkit" in preflight[0]
    assert "mkdir -p /media/nvme/llama" in preflight[0][-1]

    assert len(deploy_calls) == 1
    src, dst, rsh = deploy_calls[0]
    assert (src, dst) == (sima_dir, "sima@devkit:/media/nvme/llama")
    control_path = next(opt for opt in preflight[0] if opt.startswith("ControlPath="))
    assert rsh.startswith("ssh -o ")
    assert control_path in rsh
    assert "ControlMaster" not in rsh
    assert len(fake_run.exit_calls) == 1


def test_ipv6_destination_uses_bare_address_for_ssh(
    tmp_path: Path, fake_run: FakeRun, deploy_calls: list
) -> None:
    _write_sima_files(tmp_path / "compiled")

    deploy_lmm.deploy(tmp_path / "compiled", "sima@[2001:db8::1]:/models")

    assert "sima@2001:db8::1" in fake_run.preflight_calls[0]
    assert deploy_calls[0][1] == "sima@[2001:db8::1]:/models"
    assert fake_run.exit_calls[0][-1] == "sima@2001:db8::1"


@pytest.mark.parametrize(
    ("returncode", "stderr", "message"),
    [
        (255, "ssh: connect to host devkit port 22: Connection timed out", "Cannot reach"),
        (3, "mkdir: cannot create directory: Permission denied", "Cannot create destination"),
        (4, "", "is not writable"),
        (1, "", "failed with exit code 1"),
    ],
)
def test_failed_preflight_skips_extraction(
    tmp_path: Path, fake_run: FakeRun, deploy_calls: list,
    returncode: int, stderr: str, message: str,
) -> None:
    _write_sima_files(tmp_path / "compiled")
    fake_run.result = (returncode, "", stderr)

    with pytest.raises(RuntimeError, match=message):
        deploy_lmm.deploy(tmp_path / "compiled", "sima@devkit:/media/nvme/llama")

    assert deploy_calls == []
    assert len(fake_run.exit_calls) == 1


def test_connection_closed_when_transfer_fails(
    tmp_path: Path, fake_run: FakeRun, monkeypatch
) -> None:
    _write_sima_files(tmp_path / "compiled")

    def interrupted(src, dst, rsh=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(deploy_lmm, "_deploy_sima_files", interrupted)

    with pytest.raises(KeyboardInterrupt):
        deploy_lmm.deploy(tmp_path / "compiled", "sima@devkit:/media/nvme/llama")

    assert len(fake_run.exit_calls) == 1


def test_rsync_rsh_ssh_options_share_connection(
    tmp_path: Path, fake_run: FakeRun, deploy_calls: list, monkeypatch
) -> None:
    _write_sima_files(tmp_path / "compiled")
    monkeypatch.setenv("RSYNC_RSH", "ssh -p 2222 -i ~/.ssh/modalix")

    deploy_lmm.deploy(tmp_path / "compiled", "devkit:/models")

    preflight = fake_run.preflight_calls[0]
    assert preflight[:5] == ["ssh", "-p", "2222", "-i", "~/.ssh/modalix"]
    assert "ControlMaster=auto" in preflight
    control_path = next(opt for opt in preflight if opt.startswith("ControlPath="))
    rsh = deploy_calls[0][2]
    assert rsh.startswith("ssh -p 2222 -i ")
    assert control_path in rsh
    assert len(fake_run.exit_calls) == 1


@pytest.mark.parametrize(
    "rsync_rsh", ["sshpass -p secret ssh", "ssh -o ControlPath=/tmp/user-master"]
)
def test_rsync_rsh_without_sharing_is_used_unchanged(
    tmp_path: Path, fake_run: FakeRun, deploy_calls: list, monkeypatch, rsync_rsh: str
) -> None:
    _write_sima_files(tmp_path / "compiled")
    monkeypatch.setenv("RSYNC_RSH", rsync_rsh)

    deploy_lmm.deploy(tmp_path / "compiled", "devkit:/models")

    assert fake_run.preflight_calls[0][:-2] == rsync_rsh.split()
    assert deploy_calls[0][2] == rsync_rsh
    assert fake_run.exit_calls == []


def test_local_destination_skips_ssh(
    tmp_path: Path, fake_run: FakeRun, deploy_calls: list
) -> None:
    _write_sima_files(tmp_path / "compiled")

    deploy_lmm.deploy(tmp_path / "compiled", str(tmp_path / "deployed"))

    assert fake_run.calls == []
    assert deploy_calls[0][1:] == (tmp_path / "deployed", None)


def test_speculative_preflight_checks_each_model_directory(
    tmp_path: Path, fake_run: FakeRun, deploy_calls: list
) -> None:
    source = tmp_path / "compiled"
    for name, is_draft in (("target", "false"), ("draft", "true")):
        sima_dir = _write_sima_files(source / name)
        (sima_dir / "devkit" / "vlm_config.json").write_text(
            f'{{"lm_cfg": {{"speculative_decoding_cfg": {{"is_draft": {is_draft}}}}}}}'
        )

    deploy_lmm.deploy(source, "devkit:/models/spec/")

    scripts = [cmd[-1] for cmd in fake_run.preflight_calls]
    assert len(scripts) == 2
    assert "mkdir -p /models/spec/target" in scripts[0]
    assert "mkdir -p /models/spec/draft" in scripts[1]
    assert [dst for _, dst, _ in deploy_calls] == [
        "devkit:/models/spec/target", "devkit:/models/spec/draft"
    ]


def test_home_relative_paths_keep_tilde_expansion(fake_run: FakeRun) -> None:
    with remote_destination.RemoteSession("devkit") as session:
        session.preflight("~/models/my model")
        session.preflight("~alice/models/my model")

    scripts = [cmd[-1] for cmd in fake_run.preflight_calls]
    assert "mkdir -p \"$HOME\"/'models/my model'" in scripts[0]
    assert "mkdir -p ~alice/'models/my model'" in scripts[1]


def test_lora_preflight_requires_elf_files(tmp_path: Path, fake_run: FakeRun, monkeypatch) -> None:
    copies = []
    monkeypatch.setattr(
        deploy_lora, "_copy_npy_files", lambda src, dst, rsh=None: copies.append((src, dst, rsh))
    )
    monkeypatch.setattr(deploy_lora, "_check_local_elf_dir", pytest.fail)

    deploy_lora.deploy(tmp_path, "devkit:/models/llama/")

    script = fake_run.preflight_calls[0][-1]
    assert script.splitlines()[0] == "test -d /models/llama/elf_files || exit 5"
    assert "mkdir -p /models/llama/npy_files" in script
    assert copies[0][1] == "devkit:/models/llama/npy_files"

    fake_run.result = (5, "", "")
    copies.clear()
    with pytest.raises(RuntimeError, match="elf_files does not exist on devkit"):
        deploy_lora.deploy(tmp_path, "devkit:/models/llama")
    assert copies == []


def test_lora_remote_root_destination_keeps_root(
    tmp_path: Path, fake_run: FakeRun, monkeypatch
) -> None:
    copies = []
    monkeypatch.setattr(
        deploy_lora, "_copy_npy_files", lambda src, dst, rsh=None: copies.append((src, dst, rsh))
    )

    deploy_lora.deploy(tmp_path, "devkit:/")

    script = fake_run.preflight_calls[0][-1]
    assert script.splitlines()[0] == "test -d /elf_files || exit 5"
    assert "mkdir -p /npy_files" in script
    assert copies[0][1] == "devkit:/npy_files"
