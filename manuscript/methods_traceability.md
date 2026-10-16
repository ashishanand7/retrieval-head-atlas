# Methods Implementation And Run Audit

Authoring companion to [Methods](methods.md), checked on 17 October 2026 against repository commit `7e9d10e` and the original run manifests. This is an internal traceability document, not additional main-paper prose or a new experiment. It records the evidence needed to check and later typeset the section without changing historical protocols.

## Recorded Sources

All recorded source-file hashes were compared with both the source revision in each manifest and the current checkout. All matched. This covered nine runs: two raw/chat pilots, marker confirmation, and smoke/confirmation for temporal, specificity, and multi-fact studies. There were 147 manifest-file comparisons, counting shared source files again for each run.

| Study | Recorded experiment commit | Hashed source files per run | Main output directory |
| --- | --- | ---: | --- |
| Marker confirmation and raw/chat pilots | `67c5619b554b0f091897bdc64ad2f17ca23b257c` for confirmation; pilot revisions remain in their own manifests | 9 | [confirmation_8192_chat](../artifacts_publication/confirmation_8192_chat/) |
| Temporal | `61c5eccf6da9bb4469124f024d207082c04268b7` | 16 | [temporal_confirmation_8192_chat](../artifacts_publication/temporal_confirmation_8192_chat/) |
| Specificity and boundary | `f584e775c107f9199a9eb04bb49370bacf762115` | 20 | [specificity_confirmation_8192](../artifacts_publication/specificity_confirmation_8192/) |
| Multi-fact | `fcf1cf7e9a50f040861767919f94ab13141a028f` | 24 | [multifact_confirmation_8192](../artifacts_publication/multifact_confirmation_8192/) |

The main runs share model revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, tokenizer/chat-template SHA-256 `60a5b4c18aa81b4018d0ea12fe9e05a5ddd59eb8b6d961ee3e13f91b1ce4f1ad`, bfloat16, NVIDIA A10G, Python 3.10.21, CUDA 12.1, and the package versions given in Methods. Architecture layer/head counts come from the frozen configuration and runtime layout checks, not a locally loaded pretrained model.

Important distinction: [the first runner](../scripts/run_publication_confirmation.py) loads the then-resolved checkpoint and records it; [later runners](../scripts/run_publication_temporal.py) request and check the explicit revision. Do not rewrite that history as prospective revision pinning in every run. The scientific comparison uses the observed matching revision and tokenizer fingerprint.

## Section And Equation Map

| Methods passage | Implementation and evidence | Check that must survive editing |
| --- | --- | --- |
| 3.1 Model, head indexing, group discovery | [modeling.py](../rha/modeling.py), `inspect_model_layout`; [frozen groups](../configs/publication_confirmation.json); [exploratory groups](../configs/semantic_functional_groups.csv); [readiness audit](../docs/publication_readiness_audit_2026-09-20.md) | 28 layers, 12 query/output heads, zero-based indices. H and C match the exploratory CSV exactly; C is the complement of H within `core19`. Discovery reused 40 identities. |
| 3.2 Multi-fact construction | [multifact.py](../rha/multifact.py), `identity_spec`, `body_parts`, `calibrate`, `token_metadata`, `build_examples`; [protocol](../docs/publication_multifact_protocol.md) | Four contiguous facts; all four codes occur once; distinct first digits; confirmation wording differs from smoke. Query changes only the requested name; swap changes two code spans. Far/near match lengths and token multisets. |
| 3.2 Legacy and marker construction | [prompts.py](../rha/prompts.py), `_needle_sentence`, `build_prompt_with_filler`, `calibrate_filler_words`; [confirmation.py](../rha/confirmation.py), `render_prompt`, `validate_pair`, `build_examples` | Five templates and Latin-word filler. Distractor-heavy code appears in the fact and final note. Marker removal is not length-compensated. Initial corruption does not force a first-digit stratum. |
| 3.2 Paired corruptions | [temporal.py](../rha/temporal.py), `build_examples`; [specificity.py](../rha/specificity.py), `build_examples` | Different-first changes digit one. Same-first preserves digit one and changes digit two, not necessarily every remaining digit. Structural candidate retries do not query the model. |
| 3.2 Specificity views and grid | [specificity.py](../rha/specificity.py), `make_body`, `render`, `condition_specs`; [protocol](../docs/publication_specificity_protocol.md) | Uniform separators differ from earlier prompts. Short keeps the distractor note. Raw-space appends one input space. Raw views have no sparse donor patches; ablations use clean prompts and only different-first cases. |
| Equations (1)-(2), patch site and scope | [interventions.py](../rha/interventions.py), `HeadPatcher`; [confirmation_scoring.py](../rha/confirmation_scoring.py), `checked_intervention`; [temporal_scoring.py](../rha/temporal_scoring.py), `patch_active`, `aligned_step` | Pre-`o_proj` slices, not attention weights or directly swapped KV caches. Time t=1 in prose maps to `step=0` in code. Unselected slices are not restored to baseline. |
| Cache handling and donor prefix | [generation.py](../rha/generation.py), `prefill_build_kv`, `one_step_logits`; [temporal_scoring.py](../rha/temporal_scoring.py), `prepare_prompt`, `continue_branch`, `evaluate` | 1,024-token prefill; all but last prompt token prefetched; caller rejects overlength before helper can truncate. Cache copies isolate conditions and branches. Later donors consume the actual recipient/candidate prefix, including under inactive scope steps. |
| Equation (3), reference replacement | [specificity_scoring.py](../rha/specificity_scoring.py), `capture_reference`, `intervention_step`; [specificity.py](../rha/specificity.py), `reserved_ids`; [multifact.py](../rha/multifact.py), `reserved_ids` | Reference final-query activation captured once for a case, reused across its conditions. One separate identity per study identity, matched view/family. No donor decoding or equal reference-length requirement. |
| Fixed control draws | [first runner](../scripts/run_publication_confirmation.py), `make_groups`; [interventions.py](../rha/interventions.py), `sample_layer_matched_disjoint`; selection functions in [temporal.py](../rha/temporal.py) and [specificity.py](../rha/specificity.py) | Seed `730021 + 1000 * original_group_index + draw`. Own-group exclusion only; within-draw sampling without replacement. Five draws can repeat sets. Generate original inventory before filtering. |
| Equation (4), dense calibration | [specificity_scoring.py](../rha/specificity_scoring.py), `final_residual`, `intervention_step`; [temporal runner](../scripts/run_publication_temporal.py), `assert_self_patch`; [multifact_scoring.py](../rha/multifact_scoring.py), `assert_control` | Whole vector before `model.model.norm`; active every prediction; logit max error <= 1e-5; generated IDs/stop must equal actual donor baseline. E5 checks all four score branches. |
| Equations (5)-(6), candidates and scores | [confirmation_scoring.py](../rha/confirmation_scoring.py), `score_continuation`; [temporal_scoring.py](../rha/temporal_scoring.py), `continue_branch`; [multifact.py](../rha/multifact.py), `extra_metrics`; [adapter](../rha/multifact_scoring.py), `prepare_prompt` | Float32 log-softmax, six scored digits, no EOS or leading-space target. 2/3/3/4 candidates across the four studies. Margins use summed scores; four-way rank is strict >0. |
| Equation (7), exact/content readouts | [confirmation.py](../rha/confirmation.py), `generation_metrics`; [temporal.py](../rha/temporal.py), `output_metrics`; [specificity.py](../rha/specificity.py), `boundary_metrics`; [multifact.py](../rha/multifact.py), `extra_metrics`; [multi-fact diagnostics](../scripts/diagnose_publication_multifact.py), `summarize_condition` | Strip whitespace only for exact match. Regex is digit-boundary, not word-boundary or arbitrary substring matching. New target counts exclude pre-existing targets; net changes also include losses. Exact text does not add an EOS requirement. |
| Equation (8), decomposition | [confirmation diagnostics](../scripts/diagnose_publication_confirmation.py), `row_diagnostics`; [multi-fact diagnostics](../scripts/diagnose_publication_multifact.py), `summarize_condition` | E2 recovers first/remaining change from aggregate and first-token scores; E3-E5 save all token scores. First-digit shares, if later reported, divide mean first-token change by mean summed sequence change, not mean per-example ratios. |
| Equations (9)-(10), pairing and uncertainty | [confirmation analyzer](../scripts/analyze_publication_confirmation.py), `paired_interval`; `paired_analysis` in [specificity analyzer](../scripts/analyze_publication_specificity.py) and [multi-fact analyzer](../scripts/analyze_publication_multifact.py) | Controls averaged within identity, then paired interactions; 4,000 draws, seed 730022, family-stratified replacement sampling, NumPy 2.5th/97.5th quantiles. Fixed draws retained, no multiplicity correction. |
| Diagnostic timing and endpoint status | [pilot analysis](../docs/publication_pilot_analysis_2026-10-14.md), [confirmation protocol follow-up](../docs/publication_confirmation_protocol.md), [specificity analysis](../docs/publication_specificity_analysis_2026-10-16.md), [multi-fact protocol](../docs/publication_multifact_protocol.md) | Exclusive-content diagnostic arose after pilots but before E2. First-number diagnostic is post-run for E4, pre-run for E5. Baseline-success subsets and output/slot decomposition are not substituted for frozen primary results. |

## Executed Sizes And Positions

Counts and ranges were recomputed from each manifest and `examples.jsonl`, not inferred from the 8K filename. A row is one condition evaluated on one paired case and contains several scoring/generation branches; it is not a separate identity or a single forward pass.

| Confirmation | Identities | Cases | Condition rows | Distinct reference identities / rendered reference prompts |
| --- | ---: | ---: | ---: | --- |
| Marker | 160 | 320 | 16,640 | None |
| Temporal | 40 | 80 | 3,520 | None |
| Specificity | 40 | 320 | 12,640 | 40 / 200 |
| Multi-fact | 40 | 160 | 6,880 | 40 / 80 |

Marker has 52 conditions per case: 2 baselines, 2 self-patches, and zero/patch for 4 focal plus 20 control groups. Temporal has 44: 2 baselines, 6 self-patches, and 3 scopes for 2 focal plus 10 control groups. Specificity has 55 conditions for each different-first chat case, 31 for each same-first chat case, and 29 for each raw-view case. Multi-fact has 31 conditions for value swap and 55 for query selection at each distance. Positive controls are absent from marker/temporal, present once per case in specificity/multi-fact. Multi-fact code computes clean ablations only in query-selection cases rather than duplicating them across both corruption labels.

| Confirmation view | Prompt-token range | Code-end-to-question gap |
| --- | --- | --- |
| Marker, marked | 8,220-8,222 | Not stored as a gap field in this run |
| Marker, unmarked | 8,208-8,211 | Not stored as a gap field in this run |
| Temporal, unmarked chat | 8,208-8,211 | Not stored as a gap field in this run |
| Specificity, far chat | 8,208-8,210 | 7,291-7,327 |
| Specificity, near chat | 8,208-8,210 | 24 |
| Specificity, short chat | 101-119 | 24 |
| Specificity, far raw | 8,179-8,181 | 7,291-7,327 |
| Specificity, far raw-space | 8,180-8,182 | 7,291-7,327 |
| Multi-fact, far chat | 8,153-8,192 | 7,151-7,332 across all four code spans |
| Multi-fact, near chat | 8,153-8,192 | 6-109 across all four code spans |

Full-study seed bases are 10,100,000 / 11,100,000 / 13,100,000 / 15,100,000 respectively, with family offsets of 100,000. Temporal/specificity/multi-fact smoke bases are 12,100,000 / 14,100,000 / 16,100,000. Raw/chat pilots share base 9,100,000. The smoke counts are 5/5/8 identities and 440/1,580/1,376 rows; each raw/chat pilot has 10 identities and 400 rows. References add 50,000 to the corresponding study seed. Stages are not pooled; disjoint identity schedules do not imply that no random answer value or template text can ever recur.

## Validation And Reproduction

The existing test suite exercises the implementations named above; no experiment code or recorded outputs were changed for this writing pass. Numerical audits read the original frozen artifacts and compare reconstructed summaries and contrasts. The equation-to-code review additionally checked time indexing, score averaging versus summed margins, the reference baseline for each action, and the distinction between newly induced and net target changes.

On 17 October 2026, all **105 tests passed**. The read-only calls below reproduced the saved marker paired analysis and the temporal, specificity, and multi-fact validation, summary, and paired-analysis objects exactly. Additional checks confirmed all 147 source-hash comparisons, the common recorded model/tokenizer/environment across nine runs, and the multi-fact slot balance. Local document links, math delimiters, and equation numbering were checked; final mathematical typesetting remains for the LaTeX pass.

Run the local CPU and artifact regression tests from the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

For the main artifact audits without rewriting results, call analyzer functions rather than their write-producing CLI entry points:

```python
import json
from pathlib import Path
from scripts.analyze_publication_confirmation import analyze as marker
from scripts.analyze_publication_temporal import analyze as temporal
from scripts.analyze_publication_specificity import analyze as specificity
from scripts.analyze_publication_multifact import analyze as multifact

root = Path("artifacts_publication")
directory = root / "confirmation_8192_chat"
assert marker(directory, 4000) == json.loads((directory / "paired_analysis.json").read_text())
for name, analyze in (
    ("temporal_confirmation_8192_chat", temporal),
    ("specificity_confirmation_8192", specificity),
    ("multifact_confirmation_8192", multifact),
):
    directory = root / name
    for key, value in analyze(directory).items():
        assert value == json.loads((directory / f"{key}.json").read_text())
```

The unchanged specificity analyzer requires the ignored, extracted `rows.jsonl`; on a fresh checkout, first extract its lossless archive with `gzip -dk artifacts_publication/specificity_confirmation_8192/rows.jsonl.gz` if that raw file is absent. The multi-fact analyzer reads its gzip directly and checks both compressed and logical-content hashes. Do not extract multi-fact rows for Git or rewrite old analyses merely to produce manuscript tables.

Tokenizer-regeneration evidence is retained in the [temporal](../artifacts_publication/temporal_confirmation_8192_chat_diagnostics.json), [specificity](../artifacts_publication/specificity_confirmation_8192_diagnostics.json), and [multi-fact](../artifacts_publication/multifact_confirmation_8192_diagnostics.json) diagnostic outputs and their dated analyses. It is separate from pretrained model inference. The current Methods pass verifies code, source provenance, saved data, equations, and tests; it does not claim a new GPU replication.

## Editorial Boundaries

- Methods defines the experiments that were run, not the optional second-model or answer-availability studies discussed in the publication plan.
- Preserve the distinction between checkpoint identity observed in the first run and explicit pinning in later runs.
- Preserve the nonfactorial grids, candidate-count differences, historical float32 score change relative to older exploratory scripts, and 24/64-token budgets.
- The sparse patch transfers current head-output slices. The positive control overwrites the whole final residual. Neither is separate Q/K-versus-V path patching.
- Keep actual donor reproduction, known-target match, newly induced targets, and recipient-task correctness separate in equations and captions.
- Do not imply that a self-patch was run separately on every individual head set: it uses the relevant union of heads for each case/scope.
- Do not imply GPU determinism flags or cross-platform bitwise reproducibility: the implementation seeds Python, NumPy, and PyTorch and uses greedy decoding, but does not enable `torch.use_deterministic_algorithms`.
- The four-figure plan remains in [the outline](outline.md). Methods motivates the task schematic in F1 and timing schematic in F3; no final manuscript figure or LaTeX build is produced by this draft.
