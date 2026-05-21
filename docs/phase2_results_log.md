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
