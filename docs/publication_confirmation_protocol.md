# Publication Confirmation Protocol

Protocol: `retrieval-confirmation-v1`, frozen before collecting new pretrained-model outcomes on 20 September 2026. This is a local experiment plan, not a registered preregistration. Changes after inspecting results must be documented as a new protocol version.

## Scope

First establish that the existing effect survives fresh examples and marker removal, and determine what it means for generated answers. Use `Qwen/Qwen2.5-1.5B-Instruct`; do not transfer these head indices to another model. Keep the historical scripts and artifacts unchanged.

The batch retains the existing five templates and Latin-word filler. It is an example-held-out replication and prompt-format/marker diagnostic, **not** a template-held-out or natural-document evaluation.

## Frozen Design

`configs/publication_confirmation.json` contains the model layout, exact four focal groups, and seed schedule:

| Item | Fixed choice |
| --- | --- |
| Single heads | L22H7 (answer-transfer candidate), L20H7 (ablation-sensitive candidate) |
| Groups | Existing answer-address group of 3; non-address core of 16 |
| Pilot examples | 2 per variant, 10 identities total; seed base 9,100,000 |
| Confirmation examples | 32 per variant, 160 identities total; seed base 10,100,000 |
| Variant identity | Fixed order: literal, alias, paraphrase, relational, distractor-heavy; offset 100,000 per variant |
| Random controls | Same count and layer histogram as each focal group; disjoint from that group; drawn without looking at new outcomes |
| Marker conditions | Marked and unmarked versions of the same filler, fact, query, and clean/corrupt answer pair |
| Prompt formats | Raw and tokenizer chat template in the pilot; record the chosen confirmation format before running its reserved examples |
| Intervention | One final-prompt-token `o_proj`-input ablation or patch; none during prefill or subsequent decoding |
| Corruption | Replace only the six-digit answer, requiring equal token lengths and a stable scoring continuation |
| Generation | Greedy, up to 24 new tokens or model EOS; retain token IDs, decoded output, and stop reason |

Pilot and confirmation seeds cannot overlap under the allowed sample counts. The builder also checks every historical `semantic_*.csv` for discovery overlap. Repeated marker modes, raw/chat formats, and later position/length runs reuse identities intentionally and must not be counted as additional independent examples.

Each control is disjoint from its own focal group, not necessarily from all other focal groups. Controls can therefore contain active heads and are never called inactive controls. One pilot draw is a diagnostic comparator, not a distributional significance test. Use five fixed draws per group for the confirmation budget below; intervals condition on those draws and do not estimate variability over all possible control sets.

Removing markers shortens the prompt slightly. No filler is regenerated or silently padded to compensate; actual prompt lengths and answer character locations are saved. The requested target length refers to calibration of the legacy marked raw prompt. Chat formatting changes its realized length, which is also saved. Any length over the explicit or model position budget fails rather than truncates.

## Readouts And Comparisons

Primary measurement pair: the two **single heads**, avoiding head-count confounding. Compare their patch deltas and their ablation deltas on the same examples. For the 3-head and 16-head groups, compare each intervention with its own count/layer-matched random controls. Do not attribute a larger 16-head ablation solely to role rather than intervention size.

Every condition records:

- Mean clean-answer token log-probability and its paired intervention delta, using float32 log-softmax.
- Full-sequence clean-minus-corrupt answer log-likelihood margin and its change. This is a candidate preference score, not open-ended accuracy.
- First clean-answer-token log-probability and clean/corrupt first-token margin. The latter is uninformative when the first tokens coincide; this is recorded explicitly.
- Greedy exact six-digit match to both candidate answers after whitespace trimming. A clean answer produced in a corrupt context is **answer transfer**, not correct answering of that corrupt document.
- Whether output contains only six digits, and separately whether it contains exactly one six-digit candidate matching the clean or corrupt answer. The extraction diagnostic must not be substituted for strict exact match silently.
- Decoded output and token-budget/EOS termination for manual failure inspection.

Clean zero-ablation is compared with the clean baseline. Clean-to-corrupt patches are compared with the corrupt baseline. Self-patches use their own clean or corrupt cache and must reproduce scores and generated token IDs. This identity test does not by itself prove that a mismatched donor has a meaningful effect; the causal conditions and random controls still matter.

The analysis uses within-example differences and bootstrap resampling of identities within variant. Marker-effect comparisons remain paired. The 95% intervals are pointwise descriptive intervals, not a corrected family of significance tests. Do not pool multiple run directories as independent observations. Run-level summaries also retain all five variants; the small pilot is for diagnosis, not claims about effect prevalence.

## Local Verification

The optional test environment uses the repo-local venv:

```bash
.venv/bin/python -m pip install -r requirements-confirmation.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/audit_publication_evidence.py
```

The tests use a tiny random Qwen architecture on CPU, not downloaded model weights or the main experiment. A tokenizer-only preparation path validates the real prompt/token contracts:

```bash
.venv/bin/python scripts/run_publication_confirmation.py \
  --prepare-only --prompt-format raw \
  --out-dir logs/publication_prepare_raw
```

Repeat with `--prompt-format chat` and a different output directory. The output directory must be empty or absent; existing runs are never overwritten. A prepared directory is not an experiment result and cannot be passed to the analysis script. Use a new directory for actual GPU execution.

## SageMaker Pilot

Run from the project root after syncing these files to SageMaker and activating `rha-venv`. The existing GPU environment should already have the core dependencies. Do not upgrade a working CUDA/PyTorch installation merely to match this Mac; run manifests record actual package versions. Both commands below load the model once per format:

```bash
python scripts/run_publication_confirmation.py \
  --stage pilot --n-per-variant 2 --target-tokens 4096 \
  --prompt-format raw --random-draws 1 \
  --out-dir artifacts_publication/pilot_4096_raw

python scripts/run_publication_confirmation.py \
  --stage pilot --n-per-variant 2 --target-tokens 4096 \
  --prompt-format chat --random-draws 1 \
  --out-dir artifacts_publication/pilot_4096_chat

python scripts/analyze_publication_confirmation.py artifacts_publication/pilot_4096_raw
python scripts/analyze_publication_confirmation.py artifacts_publication/pilot_4096_chat
```

Each format has 10 independent identities, 20 paired-prompt cases including marker modes, and 20 conditions per case: two baselines, two self-patches, and ablation/patching for four focal plus four random groups. This produces **400 condition rows per format**, not 400 independent examples. The same 10 identities occur in both formats.

Use the manifest's elapsed time and peak CUDA memory to budget subsequent experiments; no pretrained-model runtime has been measured for this new runner. The 24GB GPU remains the intended execution environment. Local execution refuses model experiments without CUDA. There is no configured remote GPU launch in this workflow yet.

## Confirmation, After Pilot Review

First record the chosen prompt format and any protocol revision in the experiment log. The following is a command template, not an instruction to run the larger batch before reviewing the pilot:

```bash
PROMPT_FORMAT=chat  # Use the format justified and recorded after pilot review.
python scripts/run_publication_confirmation.py \
  --stage confirmation --n-per-variant 32 --target-tokens 8192 \
  --needle-frac 0.1 --prompt-format "$PROMPT_FORMAT" --random-draws 5 \
  --out-dir "artifacts_publication/confirmation_8192_${PROMPT_FORMAT}"

python scripts/analyze_publication_confirmation.py \
  "artifacts_publication/confirmation_8192_${PROMPT_FORMAT}"
```

This is 160 independent identities, 320 paired-prompt cases and 52 conditions per case, or 16,640 condition rows. That is a planning budget, not a power calculation. Profile the pilot first; reduce breadth before execution if necessary and document the change. A later 16K/position comparison should retain the identity pairing and be reported as a repeated-condition test.

Do not adjust the focal heads or discard poor-performing examples after viewing confirmation results. New-template selection, query-selection corruption, local-task specificity, nonzero ablation, and second-model replication are separate planned stages, not implemented by this first runner.

## Output Contract

- `manifest.json`: prepared/running/failed/complete status, exact groups/control draws, seeds, model/config, package versions, git revision and dirty state, source hashes, model/tokenizer revisions where exposed, realized run time and CUDA memory.
- `examples.jsonl`: full paired prompts, exact answers, identity/variant, marker/format settings, prompt hashes, actual lengths and answer locations. These are synthetic prompts, not private documents.
- `rows.jsonl`: flushed after every condition, so a crash leaves inspectable partial results.
- `summary.json`: descriptive per-condition and per-variant results, only after a complete run.
- `paired_analysis.json`: paired effect contrasts and pointwise bootstrap intervals, created by the analysis command.

There is deliberately no automatic resume or silent append: run a failed attempt into a new directory after fixing the cause, keeping its partial outputs for diagnosis. Terminal logs can help with exceptions, but the files contain the scientific readouts and environment details. Fetch the entire run directories back for analysis.
