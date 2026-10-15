# Specificity Confirmation Outputs

This completed run contains 12,640 condition rows across 320 cases and 40 study
identities. All seven scientific output files are preserved. The original
152,238,787-byte `rows.jsonl` is stored losslessly as `rows.jsonl.gz` to fit
GitHub's per-file limit; the uncompressed copy is ignored by Git.

Before using the unchanged analyzer on a fresh checkout, restore the trace from
the repository root (the archive is retained):

```bash
gzip -dk artifacts_publication/specificity_confirmation_8192/rows.jsonl.gz
```

Skip extraction when `rows.jsonl` already exists. Its SHA-256 must match
`rows_sha256` in `manifest.json`; the analyzer verifies that hash. Compression
was checked by byte-for-byte comparison against the original SageMaker output.
No rows, scores, generations, or traces were removed, and the experiment's
manifest and analysis outputs were not changed for storage.
