# New-Prose Multi-Fact And Query-Selection Study

Frozen pre-outcome protocol: `retrieval-multifact-v1`, 16 October 2026. Configuration: `configs/publication_multifact.json`. This follows the [specificity analysis](publication_specificity_analysis_2026-10-16.md). Existing experiment sources and artifacts are unchanged.

## Question And Scope

The previous studies found selective answer-likelihood/onset effects without complete sparse-head answer transfer, even when the full-residual positive control reproduced donor outputs exactly. Effects also occurred with nearby facts, and raw-prompt formatting explained much of the core zero-ablation penalty. Those findings do not establish a general long-range retrieval circuit.

This bounded study asks whether the frozen head effects survive new prose and selection among several plausible facts. It is **synthetic, number-valued entity lookup**, not a natural-corpus benchmark, multi-hop reasoning test or second-model replication. Null results are informative boundaries, not a reason to keep changing heads on this evaluation set.

## Paired Task Design

Each document has four equally plausible facts from one new family: parcel dispatch, archive collections, workshop service records or room reservations. Every fact associates a different person with a six-digit code. Administrative background consists of complete English paragraphs, not the historical word filler. There are no needle markers or explicitly labelled irrelevant code sentences. Each code occurs exactly once in every document.

For illustration, a document could assign Emma `123456`, Jack `234567`, Maya `345678` and Nina `456789`. The clean question asks for Emma's code. Two corrupted recipients use that same donor and the same candidate answer pair:

| Corruption | Document change | Question change | Correct recipient code |
|---|---|---|---|
| `value_swap` | Exchange Emma's and Jack's codes | None; still asks for Emma | `234567` |
| `query_selection` | None; every fact stays fixed | Ask for Jack instead of Emma | `234567` |

The value-swap design is a deliberate refinement of the proposed value substitution: it keeps all four candidate codes present and equally frequent in both prompts. It edits two code spans, whereas query selection edits a name in the question. Their effect-size contrast is **not** a comparison at equal numbers of perturbed tokens, or a pure decomposition of selection versus transport.

Each identity has a far and near view. The same four-fact cluster moves from roughly 10% through the background to immediately before the question; its internal order is unchanged. Far and near prompts have identical token counts and token multisets. All four source spans and their actual distances to the question are saved. Whole background paragraphs are added up to the 8,192-token target without truncation; actual prompt length, not merely the target, is reported.

The target fact slot is balanced across the four positions. The alternate slot is balanced separately and always differs; the 12 ordered target/alternate pairs occur three or four times in the full study. Person aliases come from a preselected equal-token-length set, based on tokenization only, not model success. Codes have distinct first digits within each document so the candidate branches separate immediately. Resampling references use fresh identities, different codes, the same family and the corresponding far/near view; they are not paired decoding donors.

Smoke and confirmation use different, frozen fact and question wordings for every family. Both sets are new relative to the earlier experiments; the confirmation wording is held out from the GPU smoke. The small number of template families still limits generalization claims. No model outcomes are used to choose examples, aliases, codes or wording.

## Frozen Interventions

Model, revision, tokenizer fingerprint, bfloat16 precision, SDPA execution and head selection match the preceding study. The model is Qwen2.5-1.5B-Instruct at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The original group-construction routine generates all controls before filtering, preserving the exact five fixed layer/size-matched draws for each retained group.

| Panel | Groups | Scope |
|---|---|---|
| Sparse donor patch | L22H7 and the three-head answer-address group, each with five controls | Prompt only; prompt plus decoding |
| Zero/resampling | Three-head address group and 16-head non-address core, each with five controls | Prompt only, on the clean prompt |
| Self-patch | Union of all heads used in that case; clean and corrupt recipients | Both patch scopes |
| Positive transfer | Entire residual vector immediately before the model's final normalization | Every prediction |

Prompt only means the **last prompt token's computation**, which predicts the first answer token, not the whole question or the entire prefill. Combined patches additionally intervene at subsequent generated/teacher-forced positions. Donor and recipient receive the recipient branch's actual prefix. No prefill is patched. Independent caches are used for all four candidate branches and free generation.

The zero/resampling panel runs only in the `query_selection` cases. The clean prompt is identical across the two corruption types, so repeating the ablations for `value_swap` would add no distinct task condition. Resampling copies a task-matched reference query activation once; the reference may have a different prompt length, and its own ID and query position are recorded. It is not a dynamic donor with a shared generated prefix.

The positive control must reproduce the donor's **actual** greedy output, termination and all four candidate token scores. Every patched step also compares the full next-token logit vector with that step's donor, with maximum absolute tolerance `1e-5`. A donor error or formatting failure remains an error, not a failed computational calibration. Positive transfer is deliberately dense and does not itself locate a sparse circuit.

The frozen temporal/specificity intervention engines are reused unchanged. A small adapter validates all four six-token candidates and extends their prepared inputs; it does not implement another decoding loop.

## Size, Identities And Endpoints

| Stage | Study identities | Per family | Paired cases | Reference prompts | Rows | Positive controls |
|---|---:|---:|---:|---:|---:|---:|
| Smoke | 8 | 2 | 32 | 16 | 1,376 | 32 |
| Confirmation | 40 | 10 | 160 | 80 | 6,880 | 160 |

Each identity has two distances and two corruptions. A value-swap case has 31 conditions: two baselines, four self-patches, one positive control and 24 sparse patches. A query-selection case adds 24 zero/resampling conditions, for 55. Reference donors are not additional statistical units. The target sample size is a bounded follow-up budget, not a formal power calculation.

Study seed bases are 16,100,000 for smoke and 15,100,000 for confirmation; families are offset by 100,000 and references by 50,000. All study/reference schedules are disjoint. Numerical seeds are checked against earlier study and reference records regardless of their old task-family names.

The two principal endpoints are complete donor-answer transfer in free generation and the clean candidate's mean per-token log probability. The principal comparison is the three-head combined patch against the average of its five fixed controls on **far, query-selection cases**. Both endpoints must be reported: a score gain alone does not demonstrate complete transfer.

Secondary readouts include the clean candidate's summed log-probability margin over the strongest of all three competitors, strict four-way ranking, clean/corrupt/other-fact generation, unique-number content, first extracted number, verbatim formatting, reference-answer content and EOS versus budget exhaustion. The generation budget is 64 tokens. Teacher forcing scores exactly six answer tokens, without EOS, and remains separate from free generation. Exact-answer metrics allow surrounding whitespace; verbatim metrics do not. A positive four-way rank is not a claim of calibrated probability over the full vocabulary.

All four actual codes are scored. The inherited scorer label `decoy` is retained only as a compatibility key for one of the other two facts; it does **not** denote an explicitly irrelevant fact. Saved examples name both other facts in `other_fact_codes`.

Paired analysis averages the five fixed controls within each identity, then uses a 4,000-draw task-family-stratified identity bootstrap. It reports focal-minus-control effects, far-minus-near effects, query-minus-swap effects, their paired interaction, combined-minus-prompt effects and zero/resampling contrasts. Confidence intervals are pointwise, without multiplicity correction; secondary contrasts are descriptive. Repeated views, conditions and references do not increase N. Zero observed behavioral events do not imply zero population probability. Never select successful examples or control draws after observing outcomes.

## Validation And Decision Gates

The runner verifies prompt alignment, candidate continuation tokenization, exact four-code inventory, swap-only/query-only token changes, paired distance token multisets, fresh seeds and pinned tokenizer identity before loading weights. The artifact auditor reconstructs task text and identity mappings, checks hashes, full condition grids, every candidate/generation branch, prefix/position/scope/hook traces, reference provenance, score arithmetic, paired deltas and all positive/self controls. It reproduces saved summaries and paired analyses. The full run requires an audited smoke from identical experiment sources, protocol, packages, model revision and chunk size.

Local tests use randomly initialized tiny Qwen models, including a CPU bfloat16 positive-control test, independent full-prefix candidate scoring, cache/branch-order checks, end-to-end row audits and deliberate corruption/failure tests. Real pinned-tokenizer preparation covers both complete seed schedules. These checks do not substitute for a pretrained SageMaker smoke or establish baseline task success.

Pre-GPU validation on 16 October: all **95 local tests** passed (21 new tests), both prepared datasets passed the design audit, and study/reference seeds were checked against 345 earlier identities. Actual tokenization ranges were:

| Stage | Study prompt tokens | Far source-to-question gap | Near source-to-question gap |
|---|---:|---:|---:|
| Smoke | 8,165-8,192 | 7,176-7,305 | 8-107 |
| Confirmation | 8,153-8,192 | 7,151-7,332 | 6-109 |

Gaps count tokens after each fact code's end and before the question, across all four facts; these are input checks, not model outcomes.

An audit failure stops interpretation. If smoke baselines are unexpectedly poor, inspect their outputs before confirmation; retain the failed evidence and version any necessary task change before a fresh smoke. The automated gate tests computation, not a desired scientific outcome or significance threshold. Confirmation failures stay in the analysis. If these frozen heads do not generalize, narrow the paper accordingly; rediscovery requires separate discovery/confirmation data.

## SageMaker Commands

From the repository root with the existing `rha-venv` activated. No new dependency installation or CUDA upgrade is required. Do not change code/configuration during a run.

```bash
git pull --ff-only
mkdir -p logs
set -o pipefail

python -u scripts/run_publication_multifact.py \
  --stage smoke \
  --out-dir artifacts_publication/multifact_smoke_8192 \
  2>&1 | tee logs/multifact_smoke_8192.log
```

Expect `PASS: 1376 rows` and 32 passing positive-control cases. Check the printed clean/corrupt baseline counts as well. If the controls pass and the task is functioning, run confirmation:

```bash
nohup python -u scripts/run_publication_multifact.py \
  --stage confirmation \
  --smoke-dir artifacts_publication/multifact_smoke_8192 \
  --out-dir artifacts_publication/multifact_confirmation_8192 \
  > logs/multifact_confirmation_8192.log 2>&1 &

tail -f logs/multifact_confirmation_8192.log
```

Expect `PASS: 6880 rows` and 160 passing positive-control cases. Ctrl-C stops `tail`, not the background job. Confirmation has five times as many cases as smoke; use the measured smoke time as an approximate budget, allowing for loading overhead and different generations. No unmeasured GPU runtime is promised.

Every invocation requires an empty output directory. Rows are flushed individually and the manifest is checkpointed per case. Failures retain partial evidence with status `failed`; there is no automatic append/resume that could mix versions. A hard process kill may leave status `running`, which the auditor also refuses.

## Outputs And Local Reproduction

Fetch both complete run directories, including these seven files:

- `manifest.json`: frozen protocols/groups, code/input hashes, environment, counts, timing and smoke provenance.
- `examples.jsonl`: all paired prompts, four candidates, changed tokens and source/query spans.
- `references.jsonl`: separate resampling prompts and provenance.
- `rows.jsonl.gz`: lossless compressed scores, generations and complete per-step traces.
- `validation.json`: grid/design/trace/control audits and baseline counts.
- `summary.json`: condition and task-family readouts.
- `paired_analysis.json`: paired identity-level effects and pointwise intervals.

Do not extract `rows.jsonl.gz` for Git. The new analyzer reads it directly and verifies both the archive hash and the uncompressed JSONL hash. Logs under ignored `logs/` are useful for exceptions but do not contain additional scientific endpoints.

```bash
python scripts/analyze_publication_multifact.py \
  artifacts_publication/multifact_confirmation_8192

# Local tests, without pretrained model weights:
.venv/bin/python -m unittest discover -s tests -v

# Optional local preparation only; use fresh output directories:
.venv/bin/python scripts/run_publication_multifact.py \
  --stage smoke --prepare-only --local-files-only \
  --out-dir logs/multifact_prepare_smoke_check

.venv/bin/python scripts/run_publication_multifact.py \
  --stage confirmation --prepare-only --local-files-only \
  --out-dir logs/multifact_prepare_confirmation_check
```

## Bounded Publication Plan

Start the manuscript once this study has been analyzed, whether its result is positive, negative or mixed. No further old-template sweep is a prerequisite for drafting. The planned next experiment batch for a stronger journal version is one targeted second-model-family replication, with separate discovery and confirmation rather than copied head indices. Its model, fit/runtime and exact jobs remain to be specified after a feasibility check.

Budget **one smoke plus one full run now**, then roughly **one second-model study** while drafting. The second-model work needs at least a feasibility/discovery step and held-out confirmation, so its number of launches is less certain than this runner's two jobs. Allow one diagnostic/retest cycle if task baselines or instrumentation reveal a real problem, not repeated experiments to obtain a preferred narrative. A carefully scoped single-model manuscript can proceed without waiting indefinitely for broader replication; submission readiness still requires related-work/novelty positioning and a reproducible research artifact, not merely another favorable plot.

## Completed Run Follow-Up

Both stages were fetched in results commit `64b6380`. The [multi-fact analysis](publication_multifact_analysis_2026-10-16.md) reproduces the positive controls and frozen analyses, distinguishes induced target switching from baseline coincidences and incorrect donor outputs, compares value swap with query selection, and recommends starting the manuscript. This protocol is retained as the pre-outcome record.
