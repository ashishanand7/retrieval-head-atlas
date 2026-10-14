# Publication Pilot Analysis

Analysis date: 14 October 2026. Results commit: `c1bfa33`; experiment code commit: `02698708869484fba9ef8dd9382cbcefc7db19bd`.

## Decision

Use chat formatting for the reserved 8K confirmation, with the original head groups frozen. Keep both marker conditions and interpret marker-free behavior explicitly. Do not scale the raw-format sweep or launch the 16K/model-replication stages yet.

The pilot supports a narrower finding than the earlier report's broad support/content decomposition. L22H7 causes a reproducible-looking shift toward the donor answer in likelihood, including without needle markers. However, large raw-format ablation losses often accompany unchanged retrieved numbers, the single support candidate L20H7 is not detectably damaging in this small chat pilot, and no patch generates the complete donor answer. The next confirmation should measure these distinctions rather than assume the previous interpretation.

## Run Integrity

Both runs completed with 400 rows, 20 prompt-pair cases, and ten independent identities. The same ten identities appear in raw/chat and marked/unmarked conditions. There are two examples per variant, not 800 independent observations.

Checks passed for the complete condition grid, saved summary recomputation, source hashes against the recorded Git revision, full example-file and individual prompt hashes, row/example alignment, historical seed separation, finite scores, paired deltas, and all saved generation metrics. Self-patch score differences are exactly zero, and generated token IDs match their respective baselines. The code verifies hook invocation and rejects overlength prompts during execution. No extra terminal logs are needed to interpret these completed scientific outputs.

The runs use the same Qwen model revision, tokenizer/template hash, bfloat16 weights, head/control sets, package versions, clean/corrupt answers, and filler. Chat wraps the same underlying text with system/user/assistant structure and changes the final prompt position. This is a bundled formatting intervention, not an isolated test of one separator token.

| Run | Realized token range | Elapsed time including model load | Peak allocated CUDA memory |
| --- | ---: | ---: | ---: |
| Raw | 4,083-4,097 | 12.85 minutes | 3.62 GiB |
| Chat | 4,112-4,126 | 5.97 minutes | 3.62 GiB |

Average generated length is 10.81 tokens in raw versus 7.04 in chat. Shorter responses are consistent with some runtime difference, but model loading and other costs are included, so the entire speed difference cannot be attributed to decoding. Peak tensor allocation is not total device usage or a GPU-utilization measurement.

## Main Comparison

Each column below contains the same ten identities. Score deltas are changes in mean clean-answer token log-probability. They are not percentage-point accuracy changes.

| Readout | Raw marked | Raw unmarked | Chat marked | Chat unmarked |
| --- | ---: | ---: | ---: | ---: |
| Clean baseline strict exact answer | 0/10 | 2/10 | 10/10 | 10/10 |
| Corrupt baseline exact recipient answer | 0/10 | 1/10 | 10/10 | 10/10 |
| Clean baseline correct number without competing six-digit numbers, diagnostic | 10/10 | 10/10 | 10/10 | 10/10 |
| L22H7 patch delta | +0.3527 | +0.4184 | +0.9904 | +0.8624 |
| Address-group patch delta | +0.3757 | +0.4454 | +1.1792 | +1.1036 |
| Non-address-core patch delta | +0.0093 | +0.0253 | +0.2923 | +0.2868 |
| L20H7 ablation delta | -0.1492 | -0.1423 | +0.000022 | +0.000104 |
| Non-address-core ablation delta | -1.1185 | -0.9923 | -0.0847 | -0.1446 |
| Complete donor answer after L22H7 patch | 0/10 | 0/10 | 0/10 | 0/10 |
| Complete donor answer after address-group patch | 0/10 | 0/10 | 0/10 | 0/10 |

The correct-number diagnostic is explicitly post-hoc: it permits repeated occurrences of the same correct number, unlike the frozen `sole_number_is_clean` metric, which requires exactly one occurrence. It must remain separate from exact match and from the frozen extraction metric. It applies to the recorded output, including outputs stopped by the token budget.

## Raw Prompting Retrieves Numbers But Misaligns The Scored Continuation

Raw clean responses usually look like ` 841301.` rather than `841301`; the marked paraphrase examples repeat the correct number in an explanation. Every raw baseline begins with the correct number after an initial space. Strict exact-match failure therefore does not establish failed retrieval.

The scorer asks for the bare digit sequence immediately after `Answer with only the six digits.`. The model actually begins its raw generation with a space token. Consequently, the raw clean score is dominated by the unlikely first digit at this exact boundary: first-digit log-probabilities are around -10 to -13, while the remaining digits become very likely once the first digit is supplied by teacher forcing.

This is a continuation-alignment problem for interpreting the score as retrieval success, not a broken arithmetic computation. The new pilot confirms this problem in the raw setup; it does not directly measure all earlier 8K/16K generated behavior.

Removing the non-address core costs -1.1185 mean log-probability units in marked raw prompts, but all ten responses still begin with the correct number. All ten now hit the 24-token budget because they continue explaining. The same pattern holds for unmarked raw prompts: all ten begin with the correct number and all ten hit the budget. Formatting and stopping behavior change without an observed loss of the initial retrieved number.

For an answer of K tokens, the first-token decomposition is computed directly from saved scores:

```text
sequence_delta = K * delta_mean_logprob
first_delta = intervened_first_logprob - baseline_first_logprob
remaining_tokens_delta = sequence_delta - first_delta
```

For non-address-core ablation, the first token accounts for 99.84% of the marked raw sequence-score loss and 99.94% of the unmarked loss. This warrants a different interpretation from "the model can no longer retrieve the number."

This is not evidence that the group's entire causal effect is mere formatting. Its clean-minus-corrupt candidate-margin changes remain substantial: -2.6533/-2.6596 in marked/unmarked raw, versus -2.5990/-2.2156 in chat. Mean gold likelihood, candidate preference, exact formatting, and retrieved content measure different aspects of behavior.

## What Survives In Chat

The tokenizer-generated chat format produces the requested six digits on every clean and corrupt baseline, with and without markers. This is the reason to select it for confirmation, independently of which head effects become larger or smaller.

L22H7 improves clean-answer likelihood on all ten chat examples in each marker condition. The single layer-matched random head has mean patch effects of only +0.0014 marked and +0.0121 unmarked, compared with +0.9904 and +0.8624 for L22H7. With only one random draw, this is a useful comparator rather than an estimate of the complete control distribution.

The L22H7-versus-L20H7 patch contrast is +0.9833, with a descriptive paired bootstrap interval [0.7163, 1.2404], in marked chat. It is +0.8530 [0.6100, 1.0927] in unmarked chat. Marker removal therefore does not eliminate this pilot's likelihood-shift distinction. It also does not establish marker independence on arbitrary tasks.

The broader non-address core is no longer nearly patch-null: its chat effect is +0.2923/+0.2868, versus +0.0288/+0.0278 for its matched random group. That remains smaller than the address-group effect, but the old statement that this group cannot donate answer information is too strong.

L22H7 supplies about 84.0% and 78.1% of the address-group patch effect in marked/unmarked chat. The earlier 88-94% range should not be carried forward as a universal ratio.

L20H7 ablation has a tiny positive mean effect on the already near-ceiling clean likelihood and changes no chat answers. That does not establish that the head is unimportant in general. It does show that the old ablation ranking is not stable across these interfaces and this score.

The full non-address group still changes chat behavior:

| Condition | Strict exact match | Correct-number content diagnostic | Failure inspection |
| --- | ---: | ---: | --- |
| Marked baseline | 10/10 | 10/10 | None |
| Marked core ablation | 9/10 | 10/10 | One response says `Galen's badge number is 518172.` |
| Unmarked baseline | 10/10 | 10/10 | None |
| Unmarked core ablation | 7/10 | 9/10 | Two explanations still contain the correct number; one response is only `Galen`. |

The missing-number response is a substantive answer failure, but it does not by itself identify whether retrieval, answer-type selection, or output control failed. A matched local-task control remains important.

## Patching Shifts Likelihood Without Recovering The Complete Answer

There are no exact donor answers among any of the 160 patch rows per format, including controls. All patched clean-versus-corrupt full-answer margins remain negative: the complete recipient answer still has higher likelihood than the complete donor answer. These are repeated experimental conditions, not 320 independent trials for a binomial population claim.

Chat address-group patches sometimes change decoding, but to partial or other answers:

| Clean donor | Corrupt recipient | Patched output | Condition |
| --- | --- | --- | --- |
| 908469 | 756351 | 956351 | Unmarked address-group patch |
| 151446 | 209222 | 109222 | Unmarked address-group patch |
| 841301 | 680760 | 80760 | Marked L22H7 patch and address-group patches |

These illustrate a change in answer onset without a complete six-digit transplant. They are not successful donor-answer retrievals.

The token decomposition reinforces the distinction:

| L22H7 patch | Full sequence log-probability delta | First-token delta | Sum of remaining-token deltas |
| --- | ---: | ---: | ---: |
| Marked chat | +5.9422 | +8.8950 | -2.9528 |
| Unmarked chat | +5.1744 | +8.2954 | -3.1211 |

The score gain is concentrated at the first donor digit; the later teacher-forced donor digits become less likely in aggregate. The intervention is applied only at the final prompt token, so this result must not be generalized to patches at later decoding steps. It also does not prove that the head intrinsically encodes only one digit.

The smallest L22H7 chat effect, approximately +0.017, is the one pair sharing its first digit (`807431` versus `850734`). This single example suggests a useful future controlled comparison, not an established subgroup result.

## What We Should Confirm Next

Proceed with the already reserved 160 identities at 8K, using chat formatting, both marker conditions, four frozen focal groups, and five matched random draws per group. Preserve all examples and the original scoring/intervention definition. No head re-selection or success-only filtering.

The confirmation questions are now explicit:

1. Does L22H7 retain its donor-likelihood effect without markers, and does it exceed matched controls on new identities?
2. How much of that effect occurs at the first answer token, and how often does it change full generated answers rather than only confidence or a prefix?
3. Do the single support candidate and the 16-head group affect strict formatting, number content, and candidate preference differently?
4. Does the broader support group's nonzero chat patch effect recur, weakening a binary donor-versus-nondonor classification?

Keep exact match and the frozen sole-number metric as recorded. Use the first-token decomposition and distinct-number diagnostic as explicitly additional readouts, defined after pilot inspection and before confirmation. The two examples per variant in this pilot are insufficient for reliable claims about differences among semantic task families. Current bootstrap intervals are descriptive, pointwise, and conditional on the sampled controls.

On SageMaker, from the project root with `rha-venv` active:

```bash
python scripts/run_publication_confirmation.py \
  --stage confirmation --n-per-variant 32 --target-tokens 8192 \
  --needle-frac 0.1 --prompt-format chat --random-draws 5 \
  --out-dir artifacts_publication/confirmation_8192_chat

python scripts/analyze_publication_confirmation.py \
  artifacts_publication/confirmation_8192_chat
```

This is 320 paired-prompt cases and 16,640 condition rows. At the observed chat pilot's average cost per condition, the same workload at 4K would take about 4.1 hours; 8K prefill is more expensive, and model-load overhead does not scale linearly. Budget several hours and measure the actual run rather than promising a precise finish time. Keep the terminal job persistent and retain its output; the runner flushes rows but does not resume automatically after an interruption.

## Publication Consequences And Later Tests

Continue the arXiv-first publication preparation, but do not publish the older broad mechanism claim unchanged. The useful distinction is between answer-likelihood influence, response-format control, and complete answer generation. This pilot exposes where those measures disagree; their disagreement alone is not a new general scientific principle or sufficient novelty for acceptance.

After the held-out chat confirmation, the next causal controls should be:

- **Answer-start alignment and local-task specificity.** Compare raw, an explicitly separated raw answer position, and chat on otherwise matched prompts; include a nearby-answer/local task. This tests whether large apparent support effects are tied to response initiation or specifically to long-distance retrieval. It changes the final token/position and should be treated as a new controlled protocol, not silently substituted into old runs.
- **Temporal intervention scope.** Compare prompt-step patches with correctly aligned later decoding-step patches. For a free-generation comparison, donors must see the same generated prefix as recipients; feeding the known donor answer as the prefix would not demonstrate free answer recovery. Include self-patch and matched random controls at each tested scope.
- **Query-selection corruption and task generality.** Follow with competing candidate facts, held-out templates/prose, and then a small second-model replication if the causal distinctions hold. Do not expand all dimensions before resolving the measurement interpretation.

The working narrative is consequently more cautious and more testable: a head that shifts the first answer decision need not transplant an entire answer, and an ablation that lowers exact-continuation likelihood need not erase the retrieved number. The 8K confirmation can establish whether these are robust findings in this model; the later controls determine their mechanism and broader relevance.

## Reproduction

```bash
.venv/bin/python scripts/analyze_publication_confirmation.py artifacts_publication/pilot_4096_raw
.venv/bin/python scripts/analyze_publication_confirmation.py artifacts_publication/pilot_4096_chat
.venv/bin/python scripts/compare_publication_pilots.py
.venv/bin/python -m unittest discover -s tests -v
```

`artifacts_publication/pilot_comparison.json` contains source hashes, integrity results, all 80 format/marker/condition summaries, 64 token decompositions, every changed output relative to its baseline, and paired chat-minus-raw contrasts. The per-run `paired_analysis.json` files retain the existing within-format comparisons. Original prompts, rows, manifests, and summaries are unchanged.
