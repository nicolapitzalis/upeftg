# Archived Code

This directory keeps research and trial code out of the active package, script,
and test paths without deleting it.

The complete pre-refactor worktree is also preserved in git:

- branch: `archive/pre-cnn-cli-refactor-20260526`
- tag: `pre-cnn-cli-refactor-20260526`

Active development now exposes the stable CNN path only:

- spectral feature extraction
- CNN layer-sequence feature aggregation
- CNN training/finalization
- checkpoint inference

Do not move historical `runs/` artifacts as part of this refactor. Existing run
metadata may contain absolute provenance paths.
