# Publication Readiness Audit

Date: 20 September 2026. Historical repository revision: `b40b5fde61288cc48dcf21e938ba095ba57e6a06`.

## Decision

Proceed with a bounded confirmation batch before drafting a submission-ready research paper. The existing artifacts support a useful exploratory case study of answer-transfer effects versus ablation sensitivity in one model. They do not yet establish a complete retrieval circuit, a generally new retrieval/support distinction, or behavioral sufficiency.

The numerical foundation checked here is intact. The priority is experimental independence, interpretation, and differentiation from prior research, rather than replacing the existing work. Keep both university reports and their artifacts unchanged as historical deliverables.

## Scope And Verification

Reviewed the capstone report, extended report/results log, original research plan, prompt/scoring/intervention implementations, frozen head definitions, figure-generation statistics, and the closest primary papers listed below. Recomputed the core six-setting patching/ablation results and the corrected interaction control from row-level CSVs. Checked identity alignment, score arithmetic, and matching clean baselines across the selected files.

The reproducible record is `docs/publication_evidence_audit_2026-09-20.json`, including SHA-256 hashes of the 20 source files. Recreate it with:

```bash
python scripts/audit_publication_evidence.py --out docs/publication_evidence_audit_2026-09-20.json
```

This is not an independent GPU replication, a complete audit of every historical notebook, or an exhaustive literature review.

Local verification completed: 15 unit/integration tests pass; Python compilation and CLI checks pass; the actual Qwen tokenizer validates 20 pilot prompt pairs in each of raw and chat formats with identical example/answer identities. Realized pilot lengths are 4,083-4,097 tokens in raw format and 4,112-4,126 in chat format across marker modes. No pretrained-model GPU experiment has been run for this new batch.

## Claim-To-Evidence Assessment

| Candidate claim | Existing evidence | Publication judgment |
| --- | --- | --- |
| L22H7 transfers answer-specific information under the tested patch | Its query-output patch increases mean clean-answer log-probability by +0.3353 to +0.5007 across six settings. | Supported as an exploratory within-model intervention result; confirm on unseen examples and report behavioral effects. |
| L22H7 carries almost all retrieval ability | Its effect is 88.5-94.3% of the tested three-head address-group patch effect. | Unsupported. The denominator is a particular group intervention, not total retrieval ability or task accuracy. |
| A broader group is more ablation-sensitive but transfers little answer information | Non-address-core ablation is -1.1980 to -1.3643; its answer-substitution patch is +0.0240 to +0.0434. | Supported for these groups, prompts, score, and patch site. A 16-head versus 3-head comparison alone does not control intervention size. Include individual-head and group-specific matched-control comparisons. |
| These groups are necessary and sufficient for retrieval | Nonzero score changes under zero ablation and partial restoration under patching. | Overstated. Use "ablation-sensitive" and "answer-transfer effect." Neither complete necessity nor behavioral sufficiency has been established. |
| Support heads are a required downstream gate for the answer patch | With non-address-core ablation, the answer-address patch still adds +0.4400 versus +0.5075 without ablation. | The tested intervention argues against a strict gate. Report this disconfirming control prominently. It does not prove that the groups never interact elsewhere. |
| The result has six independent replications | 8K/16K times early/middle/late positions. | There are 40 unique variant/seed identities across all six settings, and all overlap the discovery sweep. Describe repeated-condition stress tests, not 240 independent examples. |
| We isolated address computation from content transport | Interventions replace head-output slices immediately before `o_proj`. | Unsupported at that granularity. They do not separately manipulate attention patterns, Q/K, or V pathways. Group labels are descriptive, not proof of a causal graph. |
| The findings characterize realistic semantic retrieval | Five small template families, six-digit targets, random-word filler, explicit needle markers. | Only controlled synthetic retrieval is established. New wording alone will not establish natural-document generality. |

At 8K/early, the L22H7 effect is +0.4571 and the clean-corrupt score gap is 3.0936. The ratio of means is **14.77%**, while the mean per-example recovery fraction is **15.15%**. These are distinct statistics. Neither is answer accuracy or a probability percentage: the underlying score is mean token log-probability.

## Reviewer Risks, In Priority Order

### R1. Selection And Confirmation Reuse

The same 40 identities were used for the neighborhood sweep, group formation, single-head patching, and position/length stress tests. This is useful exploratory evidence but selection can favor these examples. Current normal-approximation intervals do not correct that reuse.

Action: freeze head groups now, reserve new pilot and confirmation seed ranges, and avoid selecting heads again on confirmation outcomes. The first batch retains old templates to isolate example/marker/format effects. Separate template-held-out validation is still required afterward.

### R2. The Task And Scoring May Not Reflect Answering Behavior

`rha/prompts.py` makes the target salient using visible `[NEEDLE_START]` and `[NEEDLE_END]` markers. The existing scripts score a six-digit continuation directly after the raw instruction, without the model's chat template. They do not record free-generated answers for the semantic runs. This is a measurement gap, not evidence that the historical numbers are fabricated or that prompt formatting necessarily invalidates their within-setting comparisons.

Action: a small raw/chat-format pilot, paired marked/unmarked documents, exact generated-answer matching, and a separately labeled permissive single-number extraction diagnostic. Inspect all outputs, including failures. Do not present a teacher-forced improvement as an accuracy improvement.

Marker removal keeps the filler and identity fixed but shortens the prompt by the marker tokens. This is a useful first robustness check, not a perfectly length-controlled attribution of marker semantics. A large effect would warrant a neutral-delimiter/equal-budget follow-up.

### R3. Corruption Choice Can Explain Weak Patching

Answer substitution leaves the query and most task structure unchanged. If a head's clean and corrupt activations are almost identical, exchanging them has little effect by construction. Existing activation-difference measurements substantiate this invariance but do not uniquely identify the computation as retrieval-specific support.

Action: complementary query-selection corruption with two candidate answers, a local/context-independent matched task, and a nonzero replacement intervention. Reuse the corrected patch-plus-ablate control rather than rerunning an uncorrected dependency story. These are later gated experiments, not included in the first runner.

### R4. Prior Work Already Covers The Broad Narrative

| Primary source | Relevant result or method | Required distinction |
| --- | --- | --- |
| [Wu et al., Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574) | Sparse retrieval heads and causal evidence for long-context factual retrieval. | A new head index or another sparse atlas is insufficient novelty. |
| [Tang et al., RazorAttention, Introduction and Section 3](https://arxiv.org/html/2407.15891v1) | Explicit retrieval/processing hypothesis, local or sink-focused non-retrieval heads, and practical cache interventions. | Do not claim to originate the retrieval/support split. Test what our paired measurements reveal that this study did not establish. |
| [Zhao et al., Understanding Synthetic Context Extension via Retrieval Heads, Section 4 and Appendix G](https://arxiv.org/html/2410.22316v4) | Retrieval heads are necessary but not sufficient for transfer; patches selected attention outputs between differently fine-tuned models across input positions. | Our proposed contrast is within one model, answer substitution, and final-prompt-token output patches. This difference is a study design, not yet proof of a useful new contribution. |
| [Zhang and Nanda, Towards Best Practices of Activation Patching, Sections 3-4](https://arxiv.org/html/2309.16042v2) | Localization depends on corruption and scoring choices. | Measurement disagreement alone is not a new general principle. Establish retrieval-specific findings, failure boundaries, and a reproducible diagnostic. |
| [Heimersheim and Nanda, How to Use and Interpret Activation Patching](https://arxiv.org/abs/2404.15255) | Interpretation depends on task, intervention, and experimental design. | Bound claims to the donor, recipient, position, and metric actually tested. |
| [Lin et al., Retrieval Heads are Dynamic](https://aclanthology.org/2026.acl-long.715/) | Retrieval-head behavior varies during generation. | A final-prompt-token case study should not be described as a fixed circuit throughout the response. |

Working contribution: a controlled empirical study of where head-removal sensitivity and answer-transfer measurements agree or disagree in retrieval, with direct behavioral checks and explicit failure boundaries. Novelty remains provisional until the complementary tests and fuller comparison support it.

### R5. Implementation And Statistical Boundaries

- `prefill_build_kv` silently truncates inputs longer than `max_len`. The new path validates lengths before calling it. Historical 16K artifacts include prompts of 16,385 tokens; the recorded 16K commands used `MAX_LEN=32768`, so this audit does **not** infer historical truncation from their length alone.
- Historical `intervention_scope="all"` affects the query and teacher-forced continuation after prefill; it does not mean every document token was ablated. The new batch uses only the final prompt token, including for generation.
- Existing hooks can silently ignore incompatible shapes or absent donors. The confirmation path validates donor/layout shapes and verifies one hook invocation per target layer.
- New likelihoods use float32 log-softmax. Historical scoring uses logits in their native dtype. Treat differences from old values as potentially numerical and distributional, not necessarily scientific change.
- Tests compare chunked prefill with full forward, teacher-forced scores with direct logits, self-patching with baseline, and branch-cache isolation on a tiny random Qwen architecture. This verifies local machinery, not pretrained-model correctness on the GPU.
- The current report figures use pointwise `mean +/- 1.96 * sd / sqrt(n)` intervals, not bootstrap intervals. The new analysis uses paired, variant-stratified bootstrap intervals and labels them pointwise and descriptive. It does not turn a battery of intervals into a multiplicity-corrected hypothesis test.

## First Experiment And Decision Rules

The exact commands, frozen groups, metrics, and output contract are in `docs/publication_confirmation_protocol.md`.

1. Run the pilot in both raw and chat formats using the same 10 new identities, with marked/unmarked versions. This tests mechanics and behavioral feasibility; it does not establish a statistically reliable effect at two examples per variant.
2. Stop for any hook, identity-patch, alignment, non-finite-score, or truncation failure. Fix the implementation and preserve failed outputs. Do not interpret an invalid run.
3. If generated answers do not usually follow the requested six-digit format, inspect outputs and token-budget stops. Compare strict matching with the separately labeled extraction diagnostic. Revise the prompt protocol before consuming reserved confirmation examples.
4. If a format solves the task well enough to study interventions, document the choice using pilot behavior and instruction compatibility, not the largest desired head effect. Keep both pilot formats in the record.
5. Run the frozen 8K confirmation on 32 new examples per variant. Preserve all examples; report variant-level baseline success and intervention effects without filtering out failures. Use a targeted 16K follow-up only after interpreting the first confirmation, preserving paired identities.
6. If ablation/patch effects reproduce only with markers or only in likelihood, narrow the claim to that boundary. If they reproduce behaviorally without markers, proceed to new-template/query-selection/task-specificity tests, then a small second-model study.

No automated arXiv release threshold is set. A defensible preprint needs a coherent contribution, credible baseline behavior, technically valid interventions, and enough robustness to support its actual claims. A negative control outcome must change the claim or experiment design, not disappear from the report.

## Manuscript Consequences

Use a separate research manuscript. Retain the historical capstone reports unchanged. Replace claims about a fully identified circuit or strict necessity/sufficiency with operational descriptions unless new evidence establishes them. Put the corrected interaction result in the main argument, not only in an appendix. Show discovery and confirmation separately, with denominators and paired sample counts visible.

The new first batch is deliberately limited: it does not yet provide new templates, natural-document evaluation, query-selection corruption, nonzero ablation, or a second model. Those remain explicit gates in the publication strategy rather than implied completed work.
