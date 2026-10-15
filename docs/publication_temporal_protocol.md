# Publication Temporal Patching Protocol

Protocol: `retrieval-temporal-v1`. Implementation dated 15 October 2026, before pretrained-model temporal outcomes. This is a repository experiment plan, not a registered preregistration. The previous confirmation scripts and outputs are unchanged.

## Question And Fixed Design

The query-only confirmation changed donor likelihood and answer onset without generating complete donor answers. This study tests whether correctly aligned interventions at subsequent decoding positions transfer the remaining answer. It does not identify a complete retrieval circuit or establish long-distance specificity.

The configuration is `configs/publication_temporal.json`. Model weights and tokenizer are pinned to Qwen revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, with the same tokenizer/chat-template fingerprint as the confirmation. GPU execution uses bfloat16 and SDPA on one CUDA device. Keep the working SageMaker environment; no new dependencies or PyTorch upgrade are required.

| Property | Smoke | Full confirmation |
| --- | ---: | ---: |
| Examples per legacy variant | 1 | 8 |
| Independent identities | 5 | 40 |
| Seed base, with the existing 100,000 variant offsets | 12,100,000 | 11,100,000 |
| Corruptions per identity | 2 | 2 |
| Prompt-pair cases | 10 | 80 |
| Conditions per case | 44 | 44 |
| Total condition rows | 440 | 3,520 |

Both stages use unmarked chat, target 8K, early answer position 0.1, and up to 24 generated tokens or EOS. The context target retains the legacy raw-marked calibration; realized unmarked-chat lengths are saved. Local preparation produced 8,209-8,210 smoke tokens and 8,208-8,211 confirmation tokens. There is no silent truncation.

Each identity has the same clean fact, query, unrelated distractor, filler, and chat wrapper in two corruption conditions:

- `different_first`: donor and recipient answer tokens differ at digit one.
- `same_first`: their first digit is identical and their second differs.

All three candidate answers have six single-digit tokens and stable prompt-continuation boundaries. Corruption changes only the clean answer's unique source span. The builder checks historical/publication identities and verifies the two new stage schedules are disjoint. Explicit repeat attempts of this same temporal protocol reuse its reserved identities; they are not additional independent data.

The two focal groups are L22H7 and the existing three-head address group. Each retains all five original layer/count-matched control draws. Group indexing is preserved before filtering so the old random seeds cannot silently change. Controls can contain other known active heads and are not called inactive.

## Intervention Semantics

The intervention replaces selected attention-head slices at `o_proj` input. Neither donor nor recipient is intervened on during document prefill.

| Scope | Final prompt token, predicting answer token 1 | Later token positions |
| --- | --- | --- |
| `prompt_only` | Patched | Unpatched |
| `decode_only` | Unpatched | Patched |
| `prompt_and_decode` | Patched | Patched |

At every intervention step, the donor is an **unpatched** forward pass in the clean context with the same prefix actually consumed by the recipient. During greedy generation the next prefix token is the recipient's argmax, never the donor's argmax or a known correct-answer token. Donor and recipient use the same token tensor and equal cache positions, with separate KV caches. The donor's current-position head outputs are captured before patching the recipient at that position.

Patching continues through free generation until EOS or the token budget, including the prediction of EOS. Teacher-forced scoring ends at the sixth candidate digit and does not include an EOS score. Each of the clean, corrupt, and unrelated-distractor scoring branches uses its own candidate prefix in both contexts, with independent caches. The free-generation branch starts from its own copy of the post-query state and never inherits a teacher-forced branch.

Prefill is performed once per clean/corrupt case context and copied for each condition. This reduces repeated long-context work without sharing mutable intervention caches. Every selected layer's patch hook must fire exactly once per active forward pass. Cache-position, prefix, hook, and scope traces are recorded; missing, extra, or misaligned execution fails the run. The original strict hook checker supplies the verified per-layer counts.

Each case contains two baselines, clean and corrupt self-patches at each of the three scopes, and 12 focal/control groups across three scopes. Self-patches must preserve per-token candidate scores, generated tokens, generated-token log-probabilities, and termination. Decoding-only patches must preserve the unpatched first prediction. All rows are retained, including baseline or intervention failures in task performance.

## Measurements And Interpretation

Rows save candidate token IDs, every teacher-forced token log-probability, decoded output, generated token IDs and log-probabilities, stop reason, and per-step traces. Candidate preferences include clean versus corrupt **and clean versus the unrelated distractor**.

Report strict exact output, the frozen sole-number extraction metric, and repeated-identical-number content diagnostics separately. A donor answer in a corrupted document is answer transfer, not improved recipient accuracy. Suffix metrics compare token positions beginning at the first differing candidate digit; they are not text extraction and can be positive when the complete answer is wrong. Report them alongside full-answer exact match, not instead of it.

Primary comparisons are full donor transfer and donor-likelihood change for each focal group versus its fixed control average, and `prompt_and_decode` versus `prompt_only` within identity. Later-only full transfer is interpretable particularly in the shared-first-digit condition; it cannot repair an already wrong first digit. The paired common-prefix design replaces the unbalanced, observational subgroup split in the previous confirmation.

Analysis resamples identities within variant, preserves pairing across scopes and corruption conditions, and averages fixed control draws within identity. Intervals are 95% pointwise, not multiplicity-adjusted, and the 40-identity batch is a diagnostic budget rather than a power calculation. A zero-width bootstrap interval from no observed transfers does not establish a zero population rate. Inspect per-variant outcomes without selecting heads or dropping failures after the run.

## Local Verification

The local test suite uses a tiny randomly initialized Qwen architecture on CPU, not the pretrained model. It checks:

- All three patched scopes and baseline scoring against an independent full-prefix, no-cache calculation that patches the corresponding positions in full-sequence activations.
- Prompt-only agreement with the previous confirmation runner.
- Self-patches at every scope, first-prediction invariance for decoding-only intervention, and branch-order independence.
- Actual donor/recipient input calls during forced test decoding: the recipient chooses a sequence that disagrees with both the gold answer and the donor's argmax, and both contexts receive that recipient sequence.
- Unchanged prepared-cache tensors, EOS/budget stopping, hook cleanup after exceptions, and rejected truncation/misalignment.
- Fixed controls, distinct seeds, paired corruption construction, complete condition grids, corrupted traces/scores, stale smoke runs, and overwrite protection.

```bash
.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python scripts/run_publication_temporal.py \
  --stage smoke --prepare-only --local-files-only \
  --out-dir logs/temporal_prepare_smoke_check

.venv/bin/python scripts/run_publication_temporal.py \
  --stage confirmation --prepare-only --local-files-only \
  --out-dir logs/temporal_prepare_confirmation_check
```

Both frozen datasets were also prepared locally with the actual pinned tokenizer. Prepared directories are not completed experiments and cannot satisfy the smoke gate. Use a new empty directory for every invocation. The pretrained CUDA experiment remains to be run on SageMaker.

## SageMaker Smoke

Run from the repository root with the existing `rha-venv` active. Pull the new code first; experiment source files must match their recorded Git commit. Untracked result directories are allowed. Do not edit experiment code during a run.

```bash
git pull --ff-only
mkdir -p logs
set -o pipefail

python -u scripts/run_publication_temporal.py \
  --stage smoke \
  --out-dir artifacts_publication/temporal_smoke_8192_chat \
  2>&1 | tee logs/temporal_smoke_8192_chat.log
```

Expected completion is `PASS: 440 rows`, followed by baseline exact counts and the output location. PASS means computational/artifact integrity, not a desired scientific effect. Each of the four baseline count entries is out of five identities; if baseline behavior is unexpectedly poor, inspect the smoke before spending on the full run. Do not filter its failures or select heads from its outcomes.

The run automatically writes and validates `summary.json`, `paired_analysis.json`, and `validation.json`. No separate analysis command is required. A failing self-patch, prefix/position/hook check, nonfinite score, incomplete grid, or source change prevents successful completion.

## SageMaker Full Run

After the smoke completes successfully, launch the full run from the same activated environment. It re-audits the smoke and checks protocol, experiment source hashes, package versions, model revision, and chunk size before loading the model. A code change requires a new smoke in a new directory. The gate does not select based on whether patching has the hoped-for effect.

```bash
nohup python -u scripts/run_publication_temporal.py \
  --stage confirmation \
  --smoke-dir artifacts_publication/temporal_smoke_8192_chat \
  --out-dir artifacts_publication/temporal_confirmation_8192_chat \
  > logs/temporal_confirmation_8192_chat.log 2>&1 &

tail -f logs/temporal_confirmation_8192_chat.log
```

`nohup` keeps the full job running when the terminal disconnects. Ctrl-C stops `tail`, not the background experiment. Expected final output is `PASS: 3520 rows`. Runtime is not estimated from the old runner: this one reuses prefills but performs additional donor/scoring forwards. Use the smoke's recorded timing as the first practical estimate; the full run has eight times as many cases, with similar per-case work and different fixed loading overhead.

The runner flushes each row and checkpoints the manifest after each case. On an exception it records `failed` and preserves partial outputs. There is no silent overwrite, append, or resume. Keep a failed run for diagnosis and use a new output directory after fixing the problem; do not delete evidence to make a command rerun.

## Outputs To Fetch

Fetch the complete smoke and confirmation directories:

- `manifest.json`: frozen protocol, exact heads, pinned revisions, source/input hashes, environment, stage, smoke provenance, status, counts, timing, peak CUDA allocation.
- `examples.jsonl`: full clean/corrupt prompts, two corruption strata, candidates, tokenization/span metadata, identity and prompt hashes.
- `rows.jsonl`: all condition scores, generations, token-level readouts, and alignment/hook traces.
- `summary.json`: descriptive per-condition, corruption-stratum, and variant summaries.
- `paired_analysis.json`: within-identity scope, control, and corruption contrasts.
- `validation.json`: complete-grid, self-patch, prefix/position/scope/hook checks and baseline counts.

Logs stay in the ignored `logs/` folder. They are useful for exceptions; complete scientific outputs are in the run directories. To rerun analysis without loading model weights:

```bash
python scripts/analyze_publication_temporal.py \
  artifacts_publication/temporal_confirmation_8192_chat
```

This experiment remains limited to the same model, legacy templates, filler, and early answer position. Answer-start/local-task specificity, new prose/templates, competing-fact/query corruption, and a second-model replication remain subsequent controls. No public submission is authorized or performed by these commands.

## Completed Run Follow-Up

Both stages were completed and fetched in results commit `b448d55`. The [temporal analysis](publication_temporal_analysis_2026-10-15.md) reproduces their saved audits and comparisons, inspects per-token scores and generated outputs, and sets the next specificity controls. The experiment protocol above is retained as the pre-outcome record; the original run files are unchanged.
