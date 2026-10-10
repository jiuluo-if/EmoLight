"""Audit repository content and Git history without echoing matched values."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys


PATTERNS = {
    "credential": re.compile(
        r"(?i)(?:(?:https?|ssh)://[^/\s:@]+:[^/\s@]+@|\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._~-]{20,})\b|"
        r"\b(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,})"
    ),
    "email": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    "absolute_path": re.compile(r"(?i)(?:\b[A-Z]:[\\/][^\s\"'<>]+|/(?:Users|home)/[^\s\"'<>]+)"),
}
INTERNAL_BASENAMES = {"agents.md", "agent.md", "task_plan.md", "findings.md", "progress.md"}
PRIVATE_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".opus"}
PRIVATE_MODEL_SUFFIXES = {".joblib", ".pkl", ".pickle", ".onnx", ".tflite", ".safetensors", ".ckpt", ".h5", ".pt", ".pth", ".npy", ".npz"}
LARGE_FILE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Finding:
    scope: str
    kind: str
    path: str
    line: int | None = None
    commit: str | None = None


def scan_content(path: str, content: str, *, scope: str = "current", commit: str | None = None) -> list[Finding]:
    findings = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        for kind, pattern in PATTERNS.items():
            if pattern.search(line):
                findings.append(Finding(scope, kind, path, line_number, commit))
    normalized_path = path.replace("\\", "/").casefold()
    if normalized_path.startswith(("reports/", "models/")) and normalized_path.endswith(".json"):
        try:
            document = json.loads(content)
        except json.JSONDecodeError:
            document = None
        if _contains_public_speaker_identifiers(document):
            findings.append(Finding(scope, "speaker_identifiers_in_report", path, commit=commit))
    return findings


def _contains_public_speaker_identifiers(value) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in {
                "speaker_id", "speaker_ids", "speaker_name", "speaker_names", "test_speaker_ids",
                "split_speaker_ids", "speaker_ids_by_split", "per_speaker_recording_aggregated",
            }:
                return True
            if _contains_public_speaker_identifiers(child):
                return True
    elif isinstance(value, list):
        return any(_contains_public_speaker_identifiers(child) for child in value)
    return False


def scan_file(path: str, file_path: Path, *, scope: str = "current", commit: str | None = None) -> list[Finding]:
    raw = file_path.read_bytes()
    if b"\0" in raw:
        return []
    return scan_content(path, raw.decode("utf-8"), scope=scope, commit=commit)


def format_findings(findings: list[Finding]) -> str:
    grouped: dict[tuple[str, str, str], dict[str, set]] = {}
    for finding in set(findings):
        group = grouped.setdefault((finding.scope, finding.kind, finding.path), {"lines": set(), "commits": set()})
        if finding.line is not None:
            group["lines"].add(finding.line)
        if finding.commit:
            group["commits"].add(finding.commit)
    rows = []
    for (scope, kind, path), metadata in sorted(grouped.items()):
        details = []
        if metadata["lines"]:
            details.append("lines=" + ",".join(map(str, sorted(metadata["lines"]))))
        if metadata["commits"]:
            details.append("commits=" + str(len(metadata["commits"])) + ", latest=" + max(metadata["commits"])[:12])
        suffix = " (" + "; ".join(details) + ")" if details else ""
        rows.append(f"{scope}: {kind}: {path}{suffix}")
    return "\n".join(rows)


def _git(root: Path, *args: str, check: bool = True) -> bytes:
    completed = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if check and completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git command failed ({args[0]}): {message}")
    return completed.stdout


def _is_internal_path(path: str) -> bool:
    normalized = path.replace("\\", "/").casefold()
    return Path(normalized).name in INTERNAL_BASENAMES or normalized.startswith("docs/superpowers/")


def _is_forbidden_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/").casefold()
    suffix = Path(normalized).suffix
    return (
        normalized.startswith(("data/private/", "recordings/", "speaker_refs/"))
        or suffix in PRIVATE_AUDIO_SUFFIXES
        or suffix in PRIVATE_MODEL_SUFFIXES
        or ("manifest" in Path(normalized).name and suffix in {".csv", ".tsv", ".json"})
    )


def _path_findings(path: str, *, scope: str, commit: str | None = None) -> list[Finding]:
    findings = []
    if _is_internal_path(path):
        findings.append(Finding(scope, "internal_file_tracked", path, commit=commit))
    if _is_forbidden_artifact(path):
        findings.append(Finding(scope, "private_artifact_tracked", path, commit=commit))
    return findings


def _check_ignore_rules(root: Path) -> list[Finding]:
    probes = (
        "AGENTS.md", "agent.md", "agents.md", "task_plan.md", "findings.md", "progress.md",
        "docs/superpowers/probe.md", "data/private/probe.wav", "data/private/probe_manifest.csv",
        "recordings/probe.wav", "speaker_refs/probe.wav", "logs/probe.log", "tmp/probe.txt",
        "experiments/probe.csv", "exports/probe.json", "probe.wav", "probe.m4a", "probe.aac",
        "probe.safetensors", "probe.tflite", "probe.ckpt", "probe.joblib", "probe.pt",
        "probe_manifest.tsv", "probe_manifest.json",
    )
    findings = []
    for path in probes:
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--no-index", "--quiet", "--", path],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            findings.append(Finding("current", "ignore_rule_missing", path))
    return findings


def scan_local_private(root: Path) -> list[Finding]:
    """Scan ignored local text artifacts for path, credential, and direct-identifier leaks."""
    roots = (root / "data" / "private", root / "logs", root / "tmp", root / "experiments", root / "exports")
    text_suffixes = {".csv", ".json", ".txt", ".log", ".md"}
    identifier_columns = {
        "email", "user_email", "username", "user_name", "user_id", "person_id", "account_id",
        "participant_id", "member_id", "full_name", "person_name", "participant_name",
        "speaker_name", "recorded_by", "device_name", "contact_name",
    }
    findings = []
    for scan_root in roots:
        if not scan_root.is_dir():
            continue
        for file_path in scan_root.rglob("*"):
            if not file_path.is_file() or file_path.suffix.casefold() not in text_suffixes:
                continue
            relative = file_path.relative_to(root).as_posix()
            findings.extend(scan_file(relative, file_path, scope="local"))
            if file_path.suffix.casefold() == ".csv":
                with file_path.open("r", encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream)
                    headers = next(reader, ())
                for header in headers:
                    if header.strip().casefold().replace("-", "_").replace(" ", "_") in identifier_columns:
                        findings.append(Finding("local", "personal_identifier_column", relative, 1))
    return findings


def _scan_paths_and_content(root: Path, paths: list[str], *, scope: str, staged: bool = False) -> list[Finding]:
    findings = []
    for path in paths:
        findings.extend(_path_findings(path, scope=scope))
        if staged:
            raw = _git(root, "show", f":{path}", check=False)
            if not raw:
                continue
            if len(raw) > LARGE_FILE_BYTES:
                findings.append(Finding(scope, "large_staged_file", path))
            if b"\0" not in raw:
                try:
                    findings.extend(scan_content(path, raw.decode("utf-8"), scope=scope))
                except UnicodeDecodeError:
                    findings.append(Finding(scope, "unscannable_text", path))
        else:
            file_path = root / Path(path)
            if not file_path.is_file():
                continue
            if file_path.stat().st_size > LARGE_FILE_BYTES:
                findings.append(Finding(scope, "large_tracked_file", path))
            try:
                findings.extend(scan_file(path, file_path, scope=scope))
            except UnicodeDecodeError:
                findings.append(Finding(scope, "unscannable_text", path))
    return findings


def scan_repository(root: Path, *, include_history: bool = True) -> tuple[list[Finding], int, int]:
    root = root.resolve()
    current_paths = [value.decode("utf-8", errors="surrogateescape") for value in _git(root, "ls-files", "-z").split(b"\0") if value]
    findings = _scan_paths_and_content(root, current_paths, scope="current")
    findings.extend(_check_ignore_rules(root))
    findings.extend(scan_local_private(root))
    config_path_value = _git(root, "rev-parse", "--git-path", "config").decode("utf-8", errors="replace").strip()
    config_path = Path(config_path_value)
    if not config_path.is_absolute():
        config_path = root / config_path
    if config_path.is_file():
        try:
            findings.extend(scan_file(".git/config", config_path, scope="local-config"))
        except UnicodeDecodeError:
            findings.append(Finding("local-config", "unscannable_text", ".git/config"))
    configured_name = _git(root, "config", "--local", "--get", "user.name", check=False)
    if configured_name.strip():
        findings.append(Finding("local-config", "git_author_identity", ".git/config"))

    staged_paths = [
        value.decode("utf-8", errors="surrogateescape")
        for value in _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMRTUXB", "-z").split(b"\0")
        if value
    ]
    findings.extend(_scan_paths_and_content(root, staged_paths, scope="staged", staged=True))

    refs = [line for line in _git(root, "for-each-ref", "--format=%(refname)").decode("utf-8", errors="replace").splitlines() if line]
    commits = [line for line in _git(root, "rev-list", "--all").decode("ascii", errors="ignore").splitlines() if line]
    if include_history:
        grep_patterns = (
            r"(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|Bearer[[:space:]]+[A-Za-z0-9._~-]{20,}|api[_-]?key[[:space:]]*[:=]|access[_-]?token[[:space:]]*[:=]|client[_-]?secret[[:space:]]*[:=])",
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            r"[A-Za-z]:[\\/][^[:space:]\"'<>]+|/(Users|home)/[^[:space:]\"'<>]+",
        )
        for commit in commits:
            tree_paths = [
                value.decode("utf-8", errors="surrogateescape")
                for value in _git(root, "ls-tree", "-r", "--name-only", "-z", commit).split(b"\0")
                if value
            ]
            for path in tree_paths:
                findings.extend(_path_findings(path, scope="history", commit=commit))
            for kind, grep_pattern in zip(PATTERNS, grep_patterns):
                output = _git(root, "grep", "-I", "-i", "-n", "-E", grep_pattern, commit, "--", check=False)
                for result_line in output.decode("utf-8", errors="replace").splitlines():
                    parts = result_line.split(":", 3)
                    if len(parts) != 4:
                        continue
                    _, path, line_number, content = parts
                    try:
                        number = int(line_number)
                    except ValueError:
                        continue
                    if PATTERNS[kind].search(content):
                        findings.append(Finding("history", kind, path, number, commit))
            speaker_fields = r"(speaker_id|speaker_ids|speaker_name|speaker_names|test_speaker_ids|split_speaker_ids|speaker_ids_by_split|per_speaker_recording_aggregated)"
            output = _git(root, "grep", "-I", "-i", "-n", "-E", speaker_fields, commit, "--", "reports", "models", check=False)
            for result_line in output.decode("utf-8", errors="replace").splitlines():
                parts = result_line.split(":", 3)
                if len(parts) == 4:
                    _, path, line_number, _ = parts
                    try:
                        number = int(line_number)
                    except ValueError:
                        continue
                    findings.append(Finding("history", "speaker_identifiers_in_report", path, number, commit))
        author_records = _git(root, "log", "--all", "--format=%H%x09%an%x09%ae").decode("utf-8", errors="replace").splitlines()
        for record in author_records:
            parts = record.split("\t", 2)
            if len(parts) == 3 and (parts[1].strip() or parts[2].strip()):
                findings.append(Finding("history", "commit_author_identity", "Git commit metadata", commit=parts[0]))
                if PATTERNS["email"].search(parts[2]):
                    findings.append(Finding("history", "commit_author_email", "Git commit metadata", commit=parts[0]))
    return sorted(set(findings), key=lambda item: (item.scope, item.kind, item.path, item.line or 0, item.commit or "")), len(refs), len(commits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check repository privacy and Git staging without printing matched values.")
    parser.add_argument("--current-only", action="store_true", help="skip historical refs for a fast local check")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        findings, ref_count, commit_count = scan_repository(root, include_history=not args.current_only)
    except (OSError, RuntimeError, UnicodeError) as error:
        print(f"Privacy audit could not complete: {type(error).__name__}", file=sys.stderr)
        return 2
    print(f"Privacy audit: refs={ref_count}, reachable_commits={commit_count}, findings={len(findings)}")
    if findings:
        print(format_findings(findings))
    blocking = [finding for finding in findings if finding.scope in {"current", "staged", "local"}]
    if blocking:
        print(f"Result: FAIL ({len(blocking)} current/staged/local finding(s)); historical findings are reported separately.")
        return 1
    historical_count = sum(finding.scope == "history" for finding in findings)
    local_config_count = sum(finding.scope == "local-config" for finding in findings)
    print(f"Result: PASS for tracked/staged/local-private content; historical findings={historical_count}, local-config findings={local_config_count}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
