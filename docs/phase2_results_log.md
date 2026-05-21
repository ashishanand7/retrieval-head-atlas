# Phase 2 Results Log

## 2026-04-26: Semantic Ablation Probe

Command family:

```bash
python scripts/run_semantic_ablation_probe.py \
  --n-per-variant 8 \
  --top-k 8 \
  --target-tokens 8192
```

Variants:

- `literal`
- `alias`
- `paraphrase`
- `relational`
- `distractor_heavy`

TopK heads came from `artifacts_new/day5_patch_single_head.csv`.

## Result Summary

Full-scope ablation:

| Variant | Baseline mean logprob | TopK delta | RandomK delta |
| --- | ---: | ---: | ---: |
| literal | -2.039 | -0.416 | +0.025 |
| alias | -1.850 | -0.376 | -0.078 |
| paraphrase | -2.171 | -0.518 | -0.042 |
| relational | -1.913 | -0.315 | -0.035 |
| distractor_heavy | -2.088 | -0.389 | -0.021 |

Query-only ablation:

| Variant | Baseline mean logprob | TopK delta | RandomK delta |
| --- | ---: | ---: | ---: |
| literal | -2.039 | -0.406 | +0.025 |
| alias | -1.850 | -0.356 | -0.075 |
| paraphrase | -2.171 | -0.518 | -0.042 |
| relational | -1.913 | -0.283 | -0.039 |
| distractor_heavy | -2.088 | -0.383 | -0.027 |

Across all examples:

- Full-scope TopK mean delta: `-0.4029`
- Full-scope RandomK mean delta: `-0.0302`
- Query-only TopK mean delta: `-0.3894`
- Query-only RandomK mean delta: `-0.0317`
- TopK hurt `40/40` examples in both runs.

## Interpretation

The discovered heads remain causally important across literal and semantic retrieval tasks. Query-only ablation preserves almost the full effect of full-scope ablation, suggesting that the main causal role is concentrated at the final query step rather than during the answer-token continuation.

This supports the next paper claim, pending stronger controls:

> The previous atlas heads are not merely literal-copy heads; they participate in a broader retrieval circuit that transfers answer-relevant information at the query position across multiple semantic retrieval formats.

## Next Control

Run the same probe with multiple layer-matched disjoint RandomK draws:

```bash
python scripts/run_semantic_ablation_probe.py \
  --n-per-variant 8 \
  --top-k 8 \
  --target-tokens 8192 \
  --n-random-draws 20 \
  --intervention-scope query \
  --out artifacts_phase2/semantic_ablation_probe_8192_n8_query_rand20.csv \
  --summary-out artifacts_phase2/semantic_ablation_probe_8192_n8_query_rand20_summary.json
```

If TopK remains separated from the RandomK draw distribution, the result is strong enough to become the first Phase 2 figure/table.

## 2026-05-18: Query-Only RandomK Distribution

Pulled artifact commit:

```text
6b3590fe819847159ad7b8af27e315661d1193b4
```

Result:

- `N=40`, `n_random_draws=20`, query-only intervention.
- Overall TopK mean delta: `-0.3894`.
- Overall RandomK draw mean delta: `-0.1588`.
- Overall TopK minus RandomK mean: `-0.2306`.
- TopK hurt `40/40` examples.
- TopK was more damaging than each example's RandomK mean for `40/40` examples.

Per variant:

| Variant | TopK delta | RandomK draw mean | Gap |
| --- | ---: | ---: | ---: |
| literal | -0.4063 | -0.1537 | -0.2526 |
| alias | -0.3563 | -0.1563 | -0.2000 |
| paraphrase | -0.5183 | -0.1454 | -0.3729 |
| relational | -0.2831 | -0.1659 | -0.1172 |
| distractor_heavy | -0.3829 | -0.1725 | -0.2104 |

Important nuance:

TopK is clearly stronger than the average RandomK draw, but it is not fully outside the RandomK distribution. The worst RandomK draw had mean delta `-0.4359`, which is more damaging than TopK. Inspection shows the random draws often sample known retrieval-neighborhood heads such as `(17,6)`, `(20,0)`, `(20,5)`, `(20,10)`, `(20,11)`, `(21,4)`, and `(21,8)`.

Interpretation:

The control pool is partially contaminated by real circuit heads. This is not a failure of the result; it suggests the semantic retrieval circuit is distributed across a broader head neighborhood than the initial TopK group.

## Next Step: Single-Head Semantic Sweep

Run a query-only single-head ablation sweep over the patch-ranked heads:

```bash
python scripts/run_semantic_single_head_sweep.py \
  --n-per-variant 8 \
  --n-heads 24 \
  --target-tokens 8192 \
  --intervention-scope query \
  --out artifacts_phase2/semantic_single_head_sweep_patch24_8192_n8_query.csv \
  --summary-out artifacts_phase2/semantic_single_head_sweep_patch24_8192_n8_query_summary.json
```

This should identify which individual heads drive the semantic variants and which RandomK draws were accidentally sampling active circuit members.

## 2026-05-18: Patch-Ranked Single-Head Semantic Sweep

Pulled artifact commit:

```text
2bf22b40ff5ed922343cc9563b6debfbe3588d76
```

Run:

- Query-only ablation.
- `N=40` examples: five variants times eight examples.
- Top 24 heads from `artifacts_new/day5_patch_single_head.csv`.
- `960` single-head rows.

Most harmful heads by mean delta:

| Rank | Head | Mean delta | Negative examples | Variant profile |
| ---: | --- | ---: | ---: | --- |
| 1 | `(21,11)` | -0.1096 | 35/40 | strongest on alias and distractor-heavy |
| 2 | `(22,7)` | -0.1067 | 37/40 | broad; strongest on distractor-heavy |
| 3 | `(18,3)` | -0.0867 | 40/40 | consistently harmful across all variants |
| 4 | `(20,9)` | -0.0548 | 35/40 | literal/paraphrase/relational |
| 5 | `(17,3)` | -0.0432 | 32/40 | paraphrase/relational |
| 6 | `(22,10)` | -0.0383 | 36/40 | broad, modest |
| 7 | `(23,4)` | -0.0360 | 40/40 | broad, modest |
| 8 | `(20,3)` | -0.0327 | 37/40 | broad, modest |

Notable non-effects or reverse effects:

- `(0,8)`, `(20,2)`, and `(21,9)` were in the earlier TopK patch-sufficiency set but were not necessary as single-head query ablations here.
- `(21,9)` had positive mean delta `+0.0322`, so ablating it slightly improved the semantic-suite logprob on average.
- The original TopK group delta was `-0.3894`; the sum of the original TopK single-head deltas is about `-0.3132`, so the group effect is not just one head and not perfectly additive.

Interpretation:

The semantic retrieval circuit has a necessary core concentrated in late-middle layers, especially `(21,11)`, `(22,7)`, `(18,3)`, `(20,9)`, `(17,3)`, `(22,10)`, `(23,4)`, and `(20,3)`. The single-head result separates patch sufficiency from ablation necessity: some heads that can restore clean behavior when patched are not individually required under semantic query ablation.

This explains the RandomK wrinkle from the previous run. Some random draws likely sampled real circuit-neighborhood heads outside the first patch-ranked TopK set.

## Next Step: Neighborhood Single-Head Sweep

Before attention/output decomposition, run one broader single-head sweep over the union of:

- the semantic single-head Top 12,
- patch-ranked Top 24,
- Day 3 far and near heads,
- non-zero Day 4 candidate heads,
- every head sampled by the 20 RandomK draws.

The head list is stored at:

```text
configs/semantic_neighborhood_heads.csv
```

Run:

```bash
python scripts/run_semantic_single_head_sweep.py \
  --heads-csv configs/semantic_neighborhood_heads.csv \
  --n-heads 61 \
  --n-per-variant 8 \
  --target-tokens 8192 \
  --intervention-scope query \
  --out artifacts_phase2/semantic_single_head_sweep_neighborhood61_8192_n8_query.csv \
  --summary-out artifacts_phase2/semantic_single_head_sweep_neighborhood61_8192_n8_query_summary.json
```

This is the final selection run before the address-vs-content decomposition. Its job is to identify the real semantic-circuit core and create a cleaner inactive-control pool.

## 2026-05-19: Neighborhood Single-Head Sweep

Pulled artifact commit:

```text
50834ed50a43d531c8ada868509f4e868e9cd3e2
```

Run:

- Query-only ablation.
- `N=40` examples: five variants times eight examples.
- `61` heads from the semantic neighborhood list.
- `2440` single-head rows.

Most harmful heads:

| Rank | Head | Mean delta | Negative examples | Variant profile |
| ---: | --- | ---: | ---: | --- |
| 1 | `(20,7)` | -0.1478 | 39/40 | broad across all five variants |
| 2 | `(21,11)` | -0.1096 | 35/40 | strongest on alias and distractor-heavy |
| 3 | `(22,7)` | -0.1067 | 37/40 | broad; strongest on distractor-heavy |
| 4 | `(18,3)` | -0.0867 | 40/40 | uniformly harmful |
| 5 | `(17,10)` | -0.0802 | 40/40 | uniformly harmful |
| 6 | `(20,8)` | -0.0747 | 37/40 | literal/paraphrase-heavy |
| 7 | `(22,0)` | -0.0726 | 40/40 | uniformly harmful |
| 8 | `(22,4)` | -0.0691 | 40/40 | uniformly harmful |
| 9 | `(21,0)` | -0.0676 | 40/40 | uniformly harmful |
| 10 | `(20,6)` | -0.0553 | 40/40 | uniformly harmful |
| 11 | `(20,9)` | -0.0548 | 35/40 | literal/paraphrase/relational |
| 12 | `(21,3)` | -0.0537 | 40/40 | uniformly harmful |
| 13 | `(21,1)` | -0.0523 | 40/40 | uniformly harmful |

Selection:

- Strong semantic core: `13` heads with mean delta below `-0.05`.
- Broader semantic core: `19` heads with mean delta below `-0.03`.
- Stored broader core at `configs/semantic_core_heads.csv`.

Interpretation:

The earlier TopK heads were a partial view. The stronger result is that semantic retrieval relies on a distributed late-middle circuit, mostly across layers `20-22`, with supporting heads in layers `17`, `18`, and `23`. The strongest newly discovered head, `(20,7)`, came from the RandomK/neighborhood search rather than the original patch-ranked TopK set. This explains why RandomK controls sometimes looked too strong: they were accidentally sampling real members of the circuit.

Sanity check:

- The 20 RandomK draw effects from the previous group-ablation run correlate strongly with the sum of this sweep's single-head effects for the sampled heads: Pearson `r = 0.92`.
- The worst RandomK draw had actual mean delta `-0.4359`; the single-head sum predicted `-0.3942`.
- This supports the interpretation that RandomK was not mysterious noise; it was sampling active circuit heads.

This is exactly the direction we wanted: the project is moving from "retrieval heads exist" to "a task-conditioned retrieval circuit has a discoverable necessary core."

## Next Step: Address Trace

The first address-vs-content decomposition run should ask whether the selected core heads attend to the source answer span at the query step.

Run:

```bash
python scripts/run_semantic_attention_trace.py \
  --heads-csv configs/semantic_core_heads.csv \
  --n-heads 19 \
  --n-per-variant 8 \
  --target-tokens 8192 \
  --out artifacts_phase2/semantic_attention_trace_core19_8192_n8.csv \
  --summary-out artifacts_phase2/semantic_attention_trace_core19_8192_n8_summary.json
```

If high-necessity heads also put high query-step attention mass on the gold or needle span, the next paper claim becomes address routing. If not, the heads may be downstream content/amplification heads, and the next intervention should focus on head output/value transport.

## 2026-05-19: Core Attention Trace

Pulled artifact commit:

```text
457c82a
```

Run:

- Query-step attention trace.
- `N=40` examples: five variants times eight examples.
- `19` heads from the broader semantic core.
- `760` trace rows.

Address-routing heads:

| Head | Gold mass | Needle mass | Argmax in needle | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `(22,7)` | 0.6648 | 0.8483 | 1.000 | direct answer-address head |
| `(21,11)` | 0.5309 | 0.7160 | 0.925 | direct answer-address head |
| `(22,10)` | 0.3553 | 0.7024 | 0.700 | direct answer/needle-address head |

Non-address necessary heads:

- `(20,7)` is the strongest necessary head, mean ablation delta `-0.1478`, but has only `0.0015` mean gold attention mass and `0.0083` needle mass.
- Many necessary heads have argmax attention on either the query tail or the first token rather than the answer span.
- Examples:
  - query-tail argmax: `(20,7)`, `(18,3)`, `(20,8)`, `(22,4)`, `(20,9)`, `(17,3)`;
  - first-token argmax: `(17,10)`, `(21,0)`, `(21,1)`, `(21,3)`, `(21,5)`, `(23,4)`.

Correlation:

- Necessity versus gold attention mass across heads: `r = 0.35`.
- Necessity versus needle attention mass across heads: `r = 0.23`.
- So direct attention to the answer explains only part of causal importance.

Interpretation:

The semantic retrieval circuit appears to have at least two functional components:

1. **Address heads** that directly route attention to the answer/needle span.
2. **Non-address core heads** that are necessary but do not directly attend to the answer; these likely perform downstream transport, residual-state preparation, query-state control, or sink/query-tail stabilization.

This is the first genuinely circuit-shaped result of Phase 2. It moves the story beyond "which heads matter" into "different heads play different roles."

## Next Step: Functional Group Ablation

Run group ablations to test whether address and non-address components are separately necessary:

```bash
python scripts/run_semantic_group_ablation.py \
  --groups-csv configs/semantic_functional_groups.csv \
  --groups answer_address,non_address_core,strong13,core19,first_token_sink,query_tail \
  --n-per-variant 8 \
  --target-tokens 8192 \
  --intervention-scope query \
  --out artifacts_phase2/semantic_group_ablation_functional_8192_n8_query.csv \
  --summary-out artifacts_phase2/semantic_group_ablation_functional_8192_n8_query_summary.json
```

If `answer_address` and `non_address_core` both hurt performance, we can frame the retrieval circuit as a two-component mechanism. The next intervention after that should be activation/output patching by component.

## 2026-05-20: Functional Group Ablation

Pulled artifact commit:

```text
67b65d8fe
```

Run:

- Query-only ablation.
- `N=40` examples: five variants times eight examples.
- Six functional groups over the semantic circuit.
- `240` group-ablation rows.

Results:

| Group | Heads | Mean delta | Negative examples | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `answer_address` | 3 | -0.1625 | 39/40 | direct address heads are necessary but not the whole circuit |
| `non_address_core` | 16 | -1.2689 | 40/40 | non-address machinery is strongly necessary |
| `strong13` | 13 | -1.2934 | 40/40 | compact high-necessity core is very damaging |
| `core19` | 19 | -1.3731 | 40/40 | broader core gives the largest total damage |
| `first_token_sink` | 6 | -0.3763 | 40/40 | sink/first-token heads are functional, not inert |
| `query_tail` | 9 | -1.0146 | 40/40 | query-tail heads are a major component |

Variant profile:

- `answer_address` hurts all variants, strongest on alias and distractor-heavy prompts.
- `non_address_core`, `strong13`, `core19`, `first_token_sink`, and `query_tail` hurt every example in every variant.
- `query_tail` is especially striking: `9` heads produce `-1.0146` mean delta, much larger than the original patch-ranked TopK group.

Interpretation:

The two-component circuit story is holding:

1. **Address heads** directly attend to the answer/needle and are causally needed.
2. **Non-address heads** do not directly attend to the answer but are much more damaging as a group, suggesting residual preparation, query-state control, transport, amplification, or sink/tail stabilization.

Caveat:

The non-address groups are larger than the address group. Raw group deltas therefore cannot be interpreted as pure per-head importance. Still, the result is not just a size artifact: the `query_tail` component has a larger per-head group effect than `answer_address`, and the single-head sweep already showed many non-address heads are individually necessary.

## Next Step: Functional Group Controls

Before making this a paper claim, run layer-matched inactive controls for each functional group. The controls are appended to:

```text
configs/semantic_functional_groups.csv
```

Run:

```bash
python scripts/run_semantic_group_ablation.py \
  --groups-csv configs/semantic_functional_groups.csv \
  --groups answer_address,answer_address_inactive_control,non_address_core,non_address_inactive_control,strong13,strong13_inactive_control,core19,core19_inactive_control,first_token_sink,first_token_sink_inactive_control,query_tail,query_tail_inactive_control \
  --n-per-variant 8 \
  --target-tokens 8192 \
  --intervention-scope query \
  --out artifacts_phase2/semantic_group_ablation_functional_controls_8192_n8_query.csv \
  --summary-out artifacts_phase2/semantic_group_ablation_functional_controls_8192_n8_query_summary.json
```

If the active functional groups beat their matched inactive controls, the next move is activation/output patching by component.

## 2026-05-20: Functional Group Inactive Controls

Pulled artifact commit:

```text
ddb27ab
```

Run:

- Query-only ablation.
- `N=40` examples: five variants times eight examples.
- Active functional groups plus layer-matched inactive controls.
- `480` group-ablation rows.

Results:

| Active group | Active delta | Control delta | Active-control gap | Worse examples |
| --- | ---: | ---: | ---: | ---: |
| `answer_address` | -0.1625 | +0.0107 | -0.1732 | 39/40 |
| `non_address_core` | -1.2689 | +0.0462 | -1.3151 | 40/40 |
| `strong13` | -1.2934 | +0.0857 | -1.3791 | 40/40 |
| `core19` | -1.3731 | +0.0572 | -1.4303 | 40/40 |
| `first_token_sink` | -0.3763 | +0.0608 | -0.4371 | 40/40 |
| `query_tail` | -1.0146 | +0.0241 | -1.0386 | 40/40 |

Interpretation:

The active functional groups decisively beat their matched controls. The inactive controls mostly improve or barely change the clean answer logprob, while active groups consistently damage it. This rules out a simple "large late-layer ablation" explanation.

Narrative status:

The Phase 2 story is now coherent:

1. The semantic retrieval circuit has a discoverable necessary core.
2. The core splits into direct answer-address heads and non-address support heads.
3. Non-address support heads, especially query-tail and first-token/sink heads, are not inert artifacts; they are causally important.
4. Matched inactive controls confirm this is not just layer, group size, or ablation volume.

This is a strong enough base to move from ablation to sufficiency.

## Next Step: Component Activation Patching

Run clean-to-corrupt query-step activation patching by component:

```bash
python scripts/run_semantic_component_patching.py \
  --groups-csv configs/semantic_functional_groups.csv \
  --groups answer_address,non_address_core,strong13,core19,first_token_sink,query_tail,answer_address_inactive_control,query_tail_inactive_control \
  --n-per-variant 8 \
  --target-tokens 8192 \
  --out artifacts_phase2/semantic_component_patching_functional_8192_n8.csv \
  --summary-out artifacts_phase2/semantic_component_patching_functional_8192_n8_summary.json
```

Expected readout:

- If `answer_address` patching helps, address heads carry sufficient source-answer information.
- If `query_tail` or `first_token_sink` patching helps, non-address components are not merely necessary stabilizers; they carry restorable causal state.
- If active components ablate strongly but patch weakly, the mechanism may be distributed or require patching multiple stages rather than query-step o-proj input alone.

## 2026-05-20: Component Activation Patching

Pulled artifact commit:

```text
b01c495
```

Run:

- Clean-to-corrupt query-step o-proj-input patching.
- `N=40` paired examples: clean and corrupt prompts differ only in the six-digit secret.
- `8` groups: six active groups plus two inactive controls.
- `320` patching rows.

Baseline:

- Clean prompt, clean gold mean logprob: `-2.0122`.
- Corrupt prompt, clean gold mean logprob: `-5.1059`.
- Mean available recovery gap: `3.0936`.
- Corrupt prompt, corrupt gold mean logprob: `-2.0432`, so the corrupt prompt successfully changes the model's expected answer.

Results:

| Group | Patch delta | Recovery | Positive examples | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `answer_address` | +0.5075 | 16.9% | 38/40 | direct answer-address heads carry recoverable answer signal |
| `core19` | +0.5161 | 17.2% | 40/40 | broader core helps only slightly more than address heads alone |
| `strong13` | +0.4746 | 15.8% | 39/40 | compact core carries recoverable signal mostly through address-containing subset |
| `non_address_core` | +0.0349 | 1.2% | 34/40 | necessary but not sufficient as clean query-step output patch |
| `query_tail` | +0.0129 | 0.4% | 20/40 | necessary, but does not itself restore answer identity |
| `first_token_sink` | +0.0109 | 0.4% | 22/40 | necessary, but not a standalone content carrier |
| `answer_address_inactive_control` | -0.0049 | -0.1% | 17/40 | inactive control does not patch |
| `query_tail_inactive_control` | -0.0014 | ~0.0% | 20/40 | inactive control does not patch |

Interpretation:

This cleanly separates **necessity** from **sufficiency**:

- Address heads are both necessary and partially sufficient. They directly attend to the answer span and clean activation patching restores a meaningful fraction of clean-answer logprob.
- Non-address groups are strongly necessary under ablation but almost not sufficient under clean-to-corrupt query-step patching. They likely provide support, gating, query-state preparation, or stabilization rather than directly carrying the answer identity.
- `core19` barely improves over `answer_address` alone, which suggests most restorable content in this patching setup lives in the address heads.

Narrative update:

The semantic retrieval circuit looks like a two-role system:

1. **Address/content heads** retrieve and carry answer-specific information.
2. **Support heads** are required for the model to use retrieval, but their activations are not themselves enough to transplant the answer.

## Next Step: Patch-Then-Ablate Interaction

Test whether non-address support heads are required to use the restored answer-address signal:

```bash
python scripts/run_semantic_patch_interaction.py \
  --groups-csv configs/semantic_functional_groups.csv \
  --patch-group answer_address \
  --ablate-groups query_tail,first_token_sink,non_address_core,query_tail_inactive_control,answer_address_inactive_control \
  --n-per-variant 8 \
  --target-tokens 8192 \
  --out artifacts_phase2/semantic_patch_interaction_answer_address_8192_n8.csv \
  --summary-out artifacts_phase2/semantic_patch_interaction_answer_address_8192_n8_summary.json
```

If ablating `query_tail`, `first_token_sink`, or `non_address_core` suppresses the benefit from answer-address patching, then the circuit is not just parallel components; the non-address support heads are needed downstream of the address heads.

## 2026-05-20: Patch-Then-Ablate Interaction

Pulled artifact commit:

```text
52696ee
```

Run:

- Patch clean `answer_address` activations into corrupt prompts.
- Then ablate candidate support groups at the same query step.
- `N=40` paired examples.
- `240` rows.

Results:

| Condition | Patch delta | Change vs patch-only | Positive patch examples |
| --- | ---: | ---: | ---: |
| `patch_only` | +0.5075 | 0.0000 | 38/40 |
| `patch_plus_ablate_first_token_sink` | +0.2323 | -0.2752 | 34/40 |
| `patch_plus_ablate_query_tail` | -0.2153 | -0.7228 | 7/40 |
| `patch_plus_ablate_non_address_core` | -0.3499 | -0.8574 | 3/40 |
| `patch_plus_ablate_query_tail_inactive_control` | +0.5329 | +0.0254 | 39/40 |
| `patch_plus_ablate_answer_address_inactive_control` | +0.5298 | +0.0223 | 38/40 |

Interpretation:

The interaction result is consistent with a dependency-chain story:

- Patching address heads helps.
- Ablating inactive controls does not suppress that help.
- Ablating `first_token_sink` weakens the patch.
- Ablating `query_tail` or the full `non_address_core` not only removes the patch benefit, but drives clean-answer logprob below the corrupt baseline.

However, one extra control is needed before making the dependency-chain claim strongly:

The current run measures `patch_plus_ablate`, but not `ablate_only` on the corrupt prompt. Without `ablate_only`, we cannot fully separate "this support group blocks use of the patch" from "this support group independently damages clean-answer logprob under the corrupt prompt."

The script has been updated to include `ablate_only_*` conditions and an explicit interaction summary.

Run:

```bash
python scripts/run_semantic_patch_interaction.py \
  --groups-csv configs/semantic_functional_groups.csv \
  --patch-group answer_address \
  --ablate-groups query_tail,first_token_sink,non_address_core,query_tail_inactive_control,answer_address_inactive_control \
  --n-per-variant 8 \
  --target-tokens 8192 \
  --out artifacts_phase2/semantic_patch_interaction_answer_address_with_ablate_only_8192_n8.csv \
  --summary-out artifacts_phase2/semantic_patch_interaction_answer_address_with_ablate_only_8192_n8_summary.json
```

Key readout:

- `patch_effect_under_ablation = patch_plus_ablate - ablate_only`.
- If this is much smaller than `patch_only - corrupt_baseline`, then the support group is genuinely required for using the address-head patch.

## 2026-05-21: Patch Interaction With Ablate-Only Controls

Pulled artifact commit:

```text
142fde0
```

Run:

- Patch clean `answer_address` activations into corrupt prompts.
- Compare `patch_only`, `ablate_only`, and `patch_plus_ablate`.
- `N=40` paired examples.
- `440` rows.

Results:

| Ablated group | Ablate-only delta | Patch under ablation | Interaction loss | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `query_tail` | -0.6628 | +0.4475 | -0.0600 | patch still works; small interaction loss |
| `first_token_sink` | -0.2674 | +0.4997 | -0.0078 | patch essentially unchanged |
| `non_address_core` | -0.7898 | +0.4400 | -0.0675 | patch still works; small interaction loss |
| `query_tail_inactive_control` | -0.0085 | +0.5414 | +0.0339 | inactive control does not suppress patch |
| `answer_address_inactive_control` | +0.0095 | +0.5203 | +0.0128 | inactive control does not suppress patch |

Interpretation:

The refined interaction control changes the dependency-chain claim:

- `query_tail` and `non_address_core` strongly affect the absolute clean-answer logprob under corrupt prompts.
- But after subtracting `ablate_only`, the answer-address patch still provides a large positive boost.
- Therefore the non-address groups are not strictly required to *use* the answer-address patch.

Updated mechanism:

The circuit is better described as a **two-role, partly additive system** rather than a strict serial chain.

1. **Address/content heads** carry answer-specific information and are partially sufficient to transplant that information.
2. **Support heads** provide an answer-independent retrieval/usefulness state. They are strongly necessary for performance, but do not themselves carry the clean answer and do not gate the address-head patch in a strict downstream sense.

This is still a strong narrative: we separated content transport from support/state machinery.

## Next Step: Activation-Difference Analysis

The next question is why support-head patching is weak despite support-head ablation being strong.

Hypothesis:

- Address heads differ substantially between clean and corrupt prompts because they encode answer identity.
- Support heads may be highly necessary but nearly invariant between clean and corrupt prompts, so clean-to-corrupt patching has little to transplant.

Run:

```bash
python scripts/run_semantic_activation_delta.py \
  --groups-csv configs/semantic_functional_groups.csv \
  --groups answer_address,non_address_core,strong13,core19,first_token_sink,query_tail,answer_address_inactive_control,query_tail_inactive_control \
  --n-per-variant 8 \
  --target-tokens 8192 \
  --include-head-rows \
  --out artifacts_phase2/semantic_activation_delta_functional_8192_n8.csv \
  --summary-out artifacts_phase2/semantic_activation_delta_functional_8192_n8_summary.json
```

If address heads show much larger clean-corrupt activation deltas than support heads, it explains the necessity/sufficiency split cleanly.

## 2026-05-21: Activation-Difference Analysis

Pulled artifact commit:

```text
fd490d2
```

Run:

- Clean-vs-corrupt query-step o-proj-input activation deltas.
- `N=40` paired examples.
- `8` groups: active functional groups plus two inactive controls.
- Includes per-head rows for all heads present in the selected groups.
- `1480` rows.

Group-level results:

| Group | Heads | Diff L2 | Relative diff | Cosine | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `answer_address` | 3 | 13.5503 | 0.8367 | 0.6423 | answer identity strongly changes the address-head activation |
| `core19` | 19 | 13.7354 | 0.4008 | 0.9151 | large because it includes address heads |
| `strong13` | 13 | 13.1641 | 0.4266 | 0.9037 | large because it includes address heads |
| `non_address_core` | 16 | 2.1159 | 0.0706 | 0.9974 | necessary support heads are nearly clean/corrupt invariant |
| `query_tail` | 9 | 1.9277 | 0.0717 | 0.9973 | necessary support heads are nearly clean/corrupt invariant |
| `first_token_sink` | 6 | 0.7716 | 0.0641 | 0.9978 | necessary support heads are nearly clean/corrupt invariant |
| `answer_address_inactive_control` | 3 | 0.5333 | 0.0672 | 0.9976 | matched inactive control is nearly invariant |
| `query_tail_inactive_control` | 9 | 1.9851 | 0.1641 | 0.9840 | activation-sensitive control, but functionally inert in prior patch/ablation |

Top per-head activation deltas:

| Head | Diff L2 | Relative diff | Cosine | Prior functional read |
| --- | ---: | ---: | ---: | --- |
| `L22H7` | 12.5690 | 1.1228 | 0.3607 | direct answer-address head |
| `L22H10` | 3.3421 | 0.6728 | 0.7460 | direct answer-address head |
| `L20H5` | 1.8128 | 0.3295 | 0.9410 | activation-sensitive but single-head ablation was null |
| `L21H11` | 2.8415 | 0.2763 | 0.9597 | direct answer-address head |
| `L22H0` | 1.0545 | 0.1630 | 0.9864 | necessary non-address/query-tail support |
| `L22H4` | 1.1247 | 0.1474 | 0.9898 | necessary non-address/query-tail support |

Interpretation:

This result strongly supports the necessity/sufficiency split:

- Address heads are the main place where changing the answer changes the query-step activation geometry.
- Support heads are causally necessary under ablation, but their query-step activations are mostly stable between clean and corrupt answers. This explains why clean-to-corrupt patching of support heads restores almost no answer probability.
- `core19` and `strong13` look activation-sensitive mainly because they include `L22H7`, `L22H10`, and `L21H11`.
- `L20H5` is an important cautionary control: it is activation-sensitive but was functionally inert in the single-head ablation sweep. That lets us say activation deltas are useful for mechanistic diagnosis, but causal intervention is still required.

Updated mechanism:

The semantic retrieval circuit now has three distinct categories:

1. **Answer-address/content heads**: direct needle attention, large clean/corrupt activation shift, partial sufficiency under patching.
2. **Support/state heads**: strong ablation necessity, low answer-specific activation shift, weak standalone patch sufficiency.
3. **Activation-sensitive bystanders**: measurable clean/corrupt activation changes without causal importance.

## Next Step: Single-Head Sufficiency Within Address Heads

Before broadening to context-position generalization, decompose the answer-address patch result into individual heads. This tests whether `L22H7` alone carries most of the recoverable answer signal, whether `L21H11` and `L22H10` contribute independently, and whether the activation-sensitive control `L20H5` patches despite being ablation-null.

A config for this has been added:

```text
configs/semantic_single_head_patch_groups.csv
```

Run:

```bash
python scripts/run_semantic_component_patching.py \
  --groups-csv configs/semantic_single_head_patch_groups.csv \
  --groups address_L22H7,address_L21H11,address_L22H10,support_L20H7,support_L18H3,support_L17H10,support_L20H8,support_L22H0,support_L22H4,activation_control_L20H5,inactive_control_L21H2,inactive_control_L22H8,inactive_control_L22H3 \
  --n-per-variant 8 \
  --target-tokens 8192 \
  --out artifacts_phase2/semantic_single_head_patching_8192_n8.csv \
  --summary-out artifacts_phase2/semantic_single_head_patching_8192_n8_summary.json
```

Expected readout:

- If `address_L22H7` recovers most of the group patch delta, the paper can name a dominant content carrier.
- If the three address heads each recover non-trivial probability, the story is an ensemble of address heads.
- If `activation_control_L20H5` has high activation delta but near-zero patch delta, it becomes a strong negative control separating activation sensitivity from causal answer transport.

After that, run the same functional suite at `--needle-frac 0.5` and `--needle-frac 0.9` to show the circuit generalizes beyond the current early-needle setting.

## 2026-05-21: Single-Head Patching

Pulled artifact commit:

```text
51d6198
```

Run:

- Clean-to-corrupt query-step o-proj-input patching.
- One head per group.
- `N=40` paired examples.
- `13` single-head groups: three answer-address heads, six support heads, one activation-sensitive control, and three inactive controls.
- `520` rows.

Baseline:

- Clean prompt, clean gold mean logprob: `-2.0122`.
- Corrupt prompt, clean gold mean logprob: `-5.1059`.
- Corrupt prompt, corrupt gold mean logprob: `-2.0432`.
- Mean available recovery gap: `3.0936`.

Results:

| Head group | Head | Patch delta | Recovery | Positive examples | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `address_L22H7` | `L22H7` | +0.4571 | 15.2% | 38/40 | dominant answer-content carrier |
| `address_L22H10` | `L22H10` | +0.0720 | 2.4% | 36/40 | smaller answer-address contributor |
| `address_L21H11` | `L21H11` | +0.0163 | 0.6% | 25/40 | attends the answer and is necessary, but weak as a standalone donor |
| `support_L22H0` | `L22H0` | +0.0054 | 0.2% | 25/40 | essentially no standalone content patch |
| `support_L22H4` | `L22H4` | +0.0015 | 0.1% | 22/40 | essentially no standalone content patch |
| `activation_control_L20H5` | `L20H5` | +0.0000 | 0.0% | 20/40 | activation-sensitive but not causal/transplantable |
| `support_L20H8` | `L20H8` | -0.0003 | 0.0% | 19/40 | no standalone content patch |
| `support_L20H7` | `L20H7` | -0.0027 | -0.1% | 18/40 | strongest single-head ablation, but not a content donor |
| `support_L18H3` | `L18H3` | -0.0079 | -0.2% | 20/40 | necessary support, not a content donor |
| `support_L17H10` | `L17H10` | -0.0118 | -0.4% | 11/40 | necessary support, not a content donor |

Address-head decomposition:

- The previous three-head `answer_address` patch delta was `+0.5075`.
- `L22H7` alone gives `+0.4571`, or about `90%` of the full address-group patch.
- `L22H10` gives `+0.0720`, useful but much smaller.
- `L21H11` gives only `+0.0163`, despite direct answer attention and a strong ablation effect.
- The three single-head deltas sum to `+0.5454`, slightly above the joint group patch, so the address heads are mildly redundant rather than perfectly additive.

Variant detail:

| Variant | Full address group | `L22H7` | `L22H10` | `L21H11` |
| --- | ---: | ---: | ---: | ---: |
| `literal` | +0.3534 | +0.2941 | +0.0249 | +0.0339 |
| `alias` | +0.6717 | +0.6152 | +0.1010 | +0.0288 |
| `paraphrase` | +0.5869 | +0.5426 | +0.0475 | +0.0296 |
| `relational` | +0.5190 | +0.4526 | +0.1607 | -0.0097 |
| `distractor_heavy` | +0.4065 | +0.3808 | +0.0261 | -0.0010 |

Interpretation:

The mechanism is now sharper than the earlier group-level story:

1. **Dominant content head**: `L22H7` is the main answer-transplant head. It has direct answer attention, the largest clean/corrupt activation shift, and nearly all of the address-group patching effect.
2. **Companion address head**: `L22H10` provides a smaller but consistent patch effect, especially for relational and alias prompts.
3. **Necessary-but-weak donor address head**: `L21H11` attends strongly to the answer and ablates strongly, but its clean activation patch barely restores the clean answer. It may help route/use the retrieved content rather than carry a cleanly transplantable value at this patch site.
4. **Support heads are not content donors**: heads such as `L20H7`, `L18H3`, and `L17H10` remain important under ablation but patch near zero as individual clean donors.
5. **Activation sensitivity is not enough**: `L20H5` has a large clean/corrupt activation delta but neither ablates nor patches meaningfully. This is a strong control separating representational change from causal answer transport.

Updated narrative:

The project can now claim a **role-decomposed retrieval circuit**:

- A dominant answer-address head (`L22H7`) carries most transplantable answer identity.
- A smaller companion address head (`L22H10`) contributes in some semantic variants.
- Other necessary heads create or preserve retrieval state but do not directly transplant the answer at the query-step o-proj input.
- Activation-only evidence is insufficient without causal patching and ablation.

## Next Step: Position Generalization

The current evidence is strong at `needle_frac=0.1`, an early-context needle. For the paper/jury story, the next question is whether the circuit is robust across context position.

Run the core suite at middle and late positions:

```bash
bash scripts/run_phase2_position_generalization.sh
```

To run only one position first:

```bash
NEEDLE_FRACS="0.5" bash scripts/run_phase2_position_generalization.sh
```

This executes, for each requested `needle_frac`:

- functional group ablation,
- functional component patching,
- single-head patching,
- attention trace,
- activation-delta analysis.

Expected readout:

- If `L22H7` remains dominant at `0.5` and `0.9`, we have a strong general semantic-retrieval circuit claim.
- If `L22H7` weakens and other heads take over, we pivot to a position-dependent circuit story.
- If support-head ablation remains strong while patching stays weak, the role-decomposition story generalizes.

## 2026-05-21: Position Generalization

Pulled artifact commit:

```text
a19a700
```

Run:

- Full phase-2 core suite at `needle_frac=0.5` and `needle_frac=0.9`.
- Compared against existing `needle_frac=0.1` results.
- `N=40` examples per position.
- `8192` target tokens.

The large diff in this commit is expected: it adds the five core artifact families for two new needle positions:

- functional group ablation,
- functional component patching,
- single-head patching,
- attention trace,
- activation-delta analysis.

### Cross-Position Summary

| Metric | `0.1` | `0.5` | `0.9` | Read |
| --- | ---: | ---: | ---: | --- |
| `answer_address` ablation | -0.1625 | -0.1262 | -0.0940 | address group remains necessary, though less than support heads |
| `non_address_core` ablation | -1.2689 | -1.3643 | -1.1980 | support/core necessity is stable and large |
| `query_tail` ablation | -1.0146 | -0.9881 | -0.9561 | query-tail support remains strongly necessary |
| `answer_address` patch | +0.5075 | +0.3729 | +0.4899 | address content patch generalizes, with a mid-position dip |
| `non_address_core` patch | +0.0349 | +0.0356 | +0.0361 | support heads still do not transplant answer identity |
| `L22H7` patch | +0.4571 | +0.3353 | +0.4619 | dominant content head remains dominant |
| `L22H10` patch | +0.0720 | +0.0580 | +0.0507 | smaller companion effect persists |
| `L21H11` patch | +0.0163 | +0.0291 | +0.0316 | weak standalone donor across positions |
| `L22H7` gold attention | 0.6648 | 0.5208 | 0.6897 | direct answer attention remains strong |
| `L22H7` needle attention | 0.8483 | 0.8053 | 0.8935 | direct needle attention remains strong |
| `L22H7` activation relative diff | 1.1228 | 1.0113 | 1.1098 | answer-specific activation shift remains very large |

Variant-level notes:

- `L22H7` stays positive for every semantic variant at every position.
- The middle-position dip is broad but not a collapse:
  - `literal`: `+0.2941 -> +0.1759 -> +0.3706`
  - `alias`: `+0.6152 -> +0.5171 -> +0.5742`
  - `paraphrase`: `+0.5426 -> +0.3006 -> +0.4550`
  - `relational`: `+0.4526 -> +0.4330 -> +0.5021`
  - `distractor_heavy`: `+0.3808 -> +0.2499 -> +0.4074`
- `L22H10` is consistently positive but smaller, with its largest relative role in relational prompts.
- `L21H11` remains weak as a clean donor despite strong attention and ablation evidence.
- Activation-sensitive control `L20H5` becomes mildly positive at later positions (`+0.0214` at `0.5`, `+0.0114` at `0.9`), but it is still far below `L22H7` and does not change the main conclusion.

Interpretation:

The position-generalization run supports the main story. This is not just an early-context artifact.

The circuit role decomposition holds across early, middle, and late needle positions:

1. **Dominant answer-content head**: `L22H7` keeps direct needle attention, large answer-specific activation shift, and the largest single-head patch effect at all positions.
2. **Companion address head**: `L22H10` remains a smaller positive donor.
3. **Support/state heads**: non-address core and query-tail groups remain highly necessary but weak as clean content donors.
4. **Causal evidence still matters**: activation-sensitive controls do not become meaningful content carriers.

The only meaningful wrinkle is the `0.5` dip in address patch strength. That is useful: the eventual paper can say the mechanism is robust but not perfectly position-invariant.

## Next Step: Context-Length Stress Test

After position generalization at `8192`, the next reviewer/jury question is whether the circuit is a genuine long-context retrieval circuit or only an 8k-context result.

The position runner now accepts `MAX_LEN` and `CHUNK_SIZE`, so it can be reused for longer targets without silent prompt truncation.

Start with a one-position 16k smoke run:

```bash
TARGET_TOKENS=16384 \
MAX_LEN=32768 \
NEEDLE_FRACS="0.5" \
N_PER_VARIANT=4 \
bash scripts/run_phase2_position_generalization.sh
```

If that succeeds, run the full 16k suite:

```bash
TARGET_TOKENS=16384 \
MAX_LEN=32768 \
NEEDLE_FRACS="0.1 0.5 0.9" \
N_PER_VARIANT=8 \
bash scripts/run_phase2_position_generalization.sh
```

Expected readout:

- If `L22H7` remains dominant at 16k, the project gets a much stronger long-context claim.
- If support necessity remains large and support patching remains weak, the role-decomposition story generalizes cleanly.
- If the identity head changes or weakens at 16k, the narrative pivots to context-length-dependent circuit reconfiguration, which is still a publishable interpretability result.
