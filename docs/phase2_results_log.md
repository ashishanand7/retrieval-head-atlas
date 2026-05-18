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
