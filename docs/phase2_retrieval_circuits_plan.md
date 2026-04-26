# Retrieval Head Atlas 2: Retrieval Circuits

## Thesis

The previous submission showed that long-context retrieval behavior in Qwen2.5-1.5B is associated with a small, structured set of attention heads, and tested those heads with ablation and activation patching. The next submission should move from "which heads light up?" to "what computation do those heads perform?"

Working claim:

> Retrieval heads are not just sparse attention artifacts. They decompose into address-routing and content-carrying subcircuits, and those subcircuits change when retrieval becomes semantic rather than literal.

This gives the jury a clean narrative: last semester mapped the atlas; this semester opens the map and identifies the circuit.

## Research Questions

1. Do high-scoring retrieval heads use attention primarily as an address lookup, a content transport path, or both?
2. Are the same heads causally important for literal copying, paraphrased retrieval, alias retrieval, and simple relational retrieval?
3. Does long-context position stress, especially middle-position stress, change the retrieval circuit?
4. Can head-level causal evidence predict when a smaller intervention such as preserving or pruning heads will preserve retrieval quality?

## Phase 1: Reproducible Spine

Goal: turn the notebook-shaped project into a scriptable research codebase.

Deliverables:

- `rha/` package with model loading, prompt generation, scoring, and head interventions.
- `scripts/` entry points that can run on SageMaker without opening notebooks.
- Stable artifact paths and CSV outputs for every experiment.
- A tiny local validation path that checks syntax and argument plumbing without requiring a GPU.

Success criterion:

- SageMaker can run a command such as `python scripts/run_semantic_ablation_probe.py --n-examples 8 --top-k 8`.
- The command writes a CSV with baseline and ablated log-probability scores.

## Phase 2: Circuit Decomposition

Goal: separate retrieval-head effects into address and content components.

Experiments:

- O-proj input ablation and patching: reproduce the earlier head-level causal result in script form.
- Attention-pattern patching: transplant attention weights while leaving value streams unchanged.
- Value/output patching: transplant head output slices while leaving attention pattern unchanged.
- Query/key stress tests: alter only the query position or the source span to check whether the head is source-addressing or answer-carrying.

Primary metrics:

- Teacher-forced gold-answer mean log-probability.
- Exact answer retrieval rate when generation is cheap enough.
- Effect size versus layer-matched disjoint random controls.
- Per-example causal effect distribution, not only aggregate means.

Expected story:

- Some heads act like source address routers.
- Some heads carry answer identity or amplify the answer once routed.
- Random controls can perform surprisingly well when they sample from the same late-layer neighborhood, which becomes a result to explain rather than an embarrassment.

## Phase 3: Beyond Literal Needles

Goal: make the project feel like interpretability, not just synthetic copy-paste.

Task families:

- Literal: the requested key is explicitly stated with the exact query wording.
- Alias: the answer is bound to an entity or codename, and the query asks through that alias.
- Paraphrase: the fact is phrased differently from the question.
- Relational: the query requires one simple bridge, such as folder -> owner -> access code.
- Distractor-heavy: many plausible numbers appear, but only one is semantically bound to the queried entity.

Primary comparison:

- Same heads across literal and semantic tasks: evidence for general retrieval heads.
- Different heads across task families: evidence for task-conditioned retrieval circuits.

## Phase 4: Long-Context Stress

Goal: connect the project to known long-context failure modes.

Experiments:

- Position sweep over early, middle, and late source locations.
- Multi-needle tasks where the model must select the correct key.
- Variable tracking or aggregation tasks inspired by RULER-style controlled benchmarks.
- A larger, deduplicated strict non-truncation LongBench slice for external validation.

Primary story:

- The atlas should explain where long-context retrieval breaks, not only where it succeeds.

## Phase 5: Paper And Portfolio Packaging

Paper structure:

1. Introduction: from retrieval heads to retrieval circuits.
2. Background: retrieval heads, activation patching, long-context evaluation.
3. Model and tasks: Qwen2.5-1.5B on A10G-scale compute.
4. Methods: mapping, ablation, patching, semantic task suite, random controls.
5. Results: head map, causal decomposition, semantic shift, long-context stress.
6. Discussion: what the circuit suggests about long-context reliability.
7. Limitations: one model family, synthetic tasks, head-level granularity, small external slice.

GitHub packaging:

- A clean README with one-command reproductions.
- Artifact tables with command, seed, model, and output path.
- Figures generated from scripts, not manually copied notebook outputs.
- A short "what changed since the previous submission" section for the jury.

## Compute Budget

Primary target: AWS SageMaker `ml.g5.xlarge` with an NVIDIA A10G-class GPU.

Practical constraints:

- Keep Qwen2.5-1.5B as the main model.
- Use small N for exploratory runs, then scale only the best experiments.
- Prefer teacher-forced log-probability before expensive generation sweeps.
- Cache prompt/example CSVs and artifacts so interrupted runs can resume cleanly.

## Immediate Next Steps

1. Scaffold reusable code around model loading, synthetic prompts, interventions, and scoring.
2. Add a first semantic ablation probe that compares baseline versus top-head ablation across literal and semantic task families.
3. Run the probe on SageMaker with small N, inspect logs, then scale.
4. Use the result to decide whether the next code investment should be attention-pattern patching or a broader semantic suite.
