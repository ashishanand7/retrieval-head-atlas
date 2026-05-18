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
