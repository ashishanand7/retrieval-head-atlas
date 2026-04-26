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
