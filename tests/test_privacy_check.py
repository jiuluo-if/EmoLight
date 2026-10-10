import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from check_privacy import format_findings, scan_content, scan_local_private


def test_sensitive_content_report_contains_location_but_not_value():
    token = "ghp" + "_" + "A" * 28
    email = "private.person" + "@example.invalid"
    local_path = "C:" + "\\" + "Users" + "\\" + "person" + "\\" + "audio.wav"

    findings = scan_content("reports/probe.txt", f"{token}\n{email}\n{local_path}")
    rendered = format_findings(findings)

    assert {finding.kind for finding in findings} == {"credential", "email", "absolute_path"}
    assert "reports/probe.txt (lines=1)" in rendered
    assert token not in rendered
    assert email not in rendered
    assert local_path not in rendered


def test_public_report_scanner_flags_speaker_identity_keys_without_echoing_values():
    speaker_id = "private-speaker-identity"
    findings = scan_content(
        "reports/evaluation.json",
        '{"test_speaker_ids": ["' + speaker_id + '"]}',
    )
    rendered = format_findings(findings)

    assert any(item.kind == "speaker_identifiers_in_report" for item in findings)
    assert speaker_id not in rendered


def test_tracked_internal_files_are_removed_and_new_names_are_ignored():
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout.decode().split("\0")
    internal_names = {"agents.md", "agent.md", "task_plan.md", "findings.md", "progress.md"}
    assert not [path for path in tracked if Path(path).name.casefold() in internal_names]
    assert all((ROOT / name).is_file() for name in ("AGENTS.md", "task_plan.md", "findings.md", "progress.md"))

    for path in (
        "AGENTS.md",
        "agent.md",
        "task_plan.md",
        "docs/superpowers/probe.md",
        "data/private/probe.csv",
        "probe_manifest.tsv",
        "probe_manifest.json",
        "recordings/probe.wav",
        "speaker_refs/probe.wav",
        "experiments/probe.csv",
        "exports/probe.json",
        "probe.aac",
        "probe.safetensors",
        "probe.tflite",
        "probe.ckpt",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", "--", path], cwd=ROOT
        )
        assert result.returncode == 0, f"expected ignore rule for {path}"


def test_unreadable_text_is_not_treated_as_a_clean_scan(tmp_path):
    from check_privacy import scan_file

    path = tmp_path / "invalid.txt"
    path.write_bytes(b"\xff\xfe\x01")

    with pytest.raises(UnicodeError):
        scan_file("reports/invalid.txt", path)


def test_local_private_manifests_are_scanned_without_echoing_paths(tmp_path):
    private_dir = tmp_path / "data" / "private"
    private_dir.mkdir(parents=True)
    manifest = private_dir / "manifest.csv"
    local_path = "D" + ":" + chr(92) + "Users" + chr(92) + "person" + chr(92) + "clip.wav"
    manifest.write_text(f"path,participant_id\n{local_path},subject-03\n", encoding="utf-8")

    findings = scan_local_private(tmp_path)
    rendered = format_findings(findings)

    assert any(item.kind == "absolute_path" for item in findings)
    assert any(item.kind == "personal_identifier_column" for item in findings)
    assert "data/private/manifest.csv" in rendered
    assert local_path not in rendered
