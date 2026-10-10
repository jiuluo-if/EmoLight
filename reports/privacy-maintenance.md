# Privacy maintenance audit

**Audit date:** 2026-10-10

**Repository line:** `main`
**Repeatable command:** `python scripts/check_privacy.py`

## Current checkout

- `AGENTS.md`, root `task_plan.md`, `findings.md`, `progress.md`, and `docs/superpowers/` were removed from the Git index with `git rm --cached`; their local files remain in place and are ignored by `.gitignore`.
- The ignore check verifies new agent-note names, temporary docs, manifests, local recordings, speaker references, logs, model weights, and experiment/export directories. The regression test also verifies the existing local files were not deleted.
- The current tracked/staged source, public reports, ignored local text artifacts, and manifest scan had no credential, email, absolute-path, private-artifact, or public speaker-identifier finding. The one local `.git/config` email match is local author configuration and is not source or staged content.
- The public model metadata and evaluation JSON contain no speaker-ID lists. They report counts and anonymous metric ranges. The model attribution and Zenodo license remain documented.
- No raw recording or personal microphone audio is tracked. EmoDB source audio, derived windows, manifests, and candidate artifacts remain under ignored `data/private/`.

## Historical Git data

At this audit pass, the scanner examined **10 local refs** and **13 reachable commits**. It reported **160 historical findings** and one separate local author-email configuration match:

- Local agent instructions and planning records previously tracked: `AGENTS.md`, `task_plan.md`, `findings.md`, `progress.md`, and `docs/superpowers/` plans/specs.
- Old absolute local paths in `findings.md` (historical lines 11, 34–35, 51), `progress.md` (line 7), and `task_plan.md` (lines 54, 59).
- Email values in old internal documents: `docs/superpowers/plans/2026-10-09-emolight-pc-mvp.md` (line 63), `findings.md` (line 13), `progress.md` (line 32), and `task_plan.md` (line 46); Git commit author names and emails are present in public metadata.
- Speaker-ID keys in earlier versions of `models/emodb_four_class.json` (line 436) and `reports/emodb_1_3_0_evaluation.json` (lines 52, 108).
- No credential-pattern match was found.

The current files are sanitized, but normal commits cannot remove old versions from Git history. No history rewrite or force-push was performed. Existing commit-author metadata and the merged PR history remain publicly addressable.

## Retention and sharing

Microphone capture is opt-in, requires GUI consent, stays local, and is not written to disk. Stopping capture clears the bounded input queue and in-memory audio window. The optional CLI prints a bounded latest-only stream and does not create an output file by default. Users can remove the locally downloaded dataset and derived outputs by manually deleting `data/private/emodb-1.3.0/`; EmoLight does not delete research data automatically.
