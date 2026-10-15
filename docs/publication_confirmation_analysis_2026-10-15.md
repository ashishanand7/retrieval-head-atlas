# Publication 8K Chat Confirmation Analysis

Analysis date: 15 October 2026. Results commit: `641d25f`; experiment source commit: `67c5619b554b0f091897bdc64ad2f17ca23b257c`. The nine recorded experiment-source hashes are identical to the pilot's source hashes.

## Decision

The confirmation supports the pilot's narrower interpretation: the selected heads can change answer likelihood and the first generated digit without transferring the complete answer. The 16-head group's ablation mostly changes response wording while preserving the retrieved number. The three-head group's ablation produces a different failure pattern: wrong answer selection on some alias questions, including selection of an explicitly unrelated distractor.

Continue the publication work, but do not describe this as a validated complete retrieval circuit or a clean separation between content heads and content-free support heads. The next implementation should be a small, correctly aligned **temporal-patching experiment**, followed by answer-start and local-task controls. Another broad context-length sweep would not resolve the main uncertainty.

This changes the order of the two immediate follow-ups in the pilot discussion: the larger chat batch makes the temporal question especially concrete. Local-task specificity remains necessary, not canceled.

## Run Integrity

The run contains **160 fresh identities**, 32 per variant, repeated in marked and unmarked conditions. There are 320 prompt-pair cases and 52 conditions per case, producing 16,640 rows. The conditions and marker repetitions are not independent examples. No identities overlap the pilot or historical discovery artifacts.

Checks passed for the complete nonduplicated grid, finite scores, exact saved-summary recomputation, recorded source hashes against Git, example-file and individual prompt hashes, metadata alignment, answer-only corruption, marker-only prompt edits, reserved seeds, and frozen control sampling. Every saved generation metric and paired delta reproduces. All self-patches have zero score error and identical generated token IDs. The supplied `paired_analysis.json` matches recomputation exactly and has not been rewritten.

An additional tokenizer-only check re-tokenized all 320 prompt pairs using the locally cached tokenizer. Its fingerprint, all saved token lengths and changed-token spans, and continuation-boundary checks match. This check does not run model weights locally.

The model revision, tokenizer/chat-template fingerprint, bfloat16 dtype, package versions, runtime configuration, and focal heads match the chat pilot. The manifest's dirty status records the newly created output directory, not modified experiment code.

| Run property | Observed value |
| --- | ---: |
| Realized marked prompt length | 8,220-8,222 tokens |
| Realized unmarked prompt length | 8,208-8,211 tokens |
| Elapsed time including model load | 6.70 hours |
| Peak allocated CUDA tensor memory | 3.69 GiB |
| Generations ending at EOS | 16,640/16,640 |
| Generations stopped by the 24-token budget | 0 |

The outputs are sufficient for the scientific analysis; terminal logs are not needed for this completed run. Memory allocation is not a GPU-utilization measurement. These timing data support budgeting another focused run, not a conclusion that the A10G was underutilized.

## Connection To The Pilots

All clean baselines answer exactly correctly: 160/160 in each marker condition. All corrupt baselines exactly reproduce their own document's replacement number: also 160/160. Thus the new behavioral failures are intervention-induced on baseline-successful examples, not baseline mistakes or truncated outputs.

All likelihood changes below are changes in **mean clean-answer token log-probability**, not accuracy percentages. The 4K and 8K columns use different examples and different numbers of random controls. They are descriptive replication comparisons, not isolated estimates of context length's causal effect.

| Readout | 4K chat marked, n=10 | 8K chat marked, n=160 | 4K chat unmarked, n=10 | 8K chat unmarked, n=160 |
| --- | ---: | ---: | ---: | ---: |
| L22H7 patch delta | +0.9904 | +1.0438 | +0.8624 | +1.0383 |
| Three-head address patch delta | +1.1792 | +1.2375 | +1.1036 | +1.2769 |
| 16-head core patch delta | +0.2923 | +0.3591 | +0.2868 | +0.3769 |
| L20H7 ablation delta | +0.000022 | +0.000127 | +0.000104 | +0.000302 |
| 16-head core ablation delta | -0.0847 | -0.0486 | -0.1446 | -0.0732 |
| Core-ablation exact clean answer | 9/10 | 141/160 | 7/10 | 137/160 |
| Core-ablation correct-number content | 10/10 | 158/160 | 9/10 | 156/160 |
| Complete donor answer after address-group patch | 0/10 | 0/160 | 0/10 | 0/160 |

The raw pilot remains important: it often produced the correct number while the bare-digit likelihood score penalized its preferred leading space. Its large core-ablation likelihood losses therefore cannot automatically be read as retrieval failures. The new confirmation shows that distinguishing likelihood, answer content, and response formatting remains necessary even after moving to a clean chat interface. It does not retrospectively measure all historical raw generations.

The content diagnostic allows repeated identical correct numbers but no competing six-digit numbers. It was defined after the pilot and before this confirmation. In this batch its counts agree with the frozen sole-number metric for these focal conditions; it is still reported separately from strict exact match.

## Likelihood Effects And Matched Controls

Each interval below is a within-identity, variant-stratified, 4,000-draw percentile bootstrap interval. Intervals are 95% pointwise, not multiplicity-adjusted. Control contrasts average the five fixed draws within each identity and condition on those draws.

| Patch group | Marked delta | Unmarked delta | Marked focal minus control mean, 95% CI | Unmarked focal minus control mean, 95% CI |
| --- | ---: | ---: | --- | --- |
| L22H7 | +1.0438 | +1.0383 | +0.9838 [0.9077, 1.0542] | +0.9691 [0.8974, 1.0379] |
| L20H7 | -0.0021 | -0.0002 | -0.0001 [-0.0039, 0.0039] | -0.0022 [-0.0050, 0.0008] |
| Address group, 3 heads | +1.2375 | +1.2769 | +1.2514 [1.1566, 1.3405] | +1.2867 [1.1935, 1.3772] |
| Non-address core, 16 heads | +0.3591 | +0.3769 | -0.2142 [-0.2394, -0.1892] | -0.2055 [-0.2281, -0.1833] |

L22H7's single-head patch contrast with L20H7 is +1.0459 [0.9661, 1.1204] marked and +1.0384 [0.9618, 1.1113] unmarked. Its mean patch effect is positive in every variant, ranging from +0.9258 to +1.1506 marked and +0.8701 to +1.2577 unmarked. Marker removal changes its overall mean by -0.0055 [-0.0224, 0.0116]. This supports persistence without explicit markers in these templates, not universal marker independence or formal equivalence.

The larger control set adds an important correction to the pilot. The core's positive patch effect replicates, but it does **not** exceed its five-draw matched-control mean. That mean is +0.5733 marked and +0.5824 unmarked, with a wide spread across fixed draws:

| Core-matched control | Address-head overlap | Marked patch delta | Unmarked patch delta |
| --- | --- | ---: | ---: |
| Draw 0 | None | +0.0273 | +0.0279 |
| Draw 1 | L21H11, L22H7 | +1.1510 | +1.1441 |
| Draw 2 | L21H11 | +0.0802 | +0.0639 |
| Draw 3 | L22H7 | +1.2179 | +1.2540 |
| Draw 4 | L21H11 | +0.3900 | +0.4220 |

The pilot used only draw 0. The other draws were generated by the same fixed sampler before new outcomes, not chosen after inspecting results. Controls were required to exclude their own focal group, not all known active heads. Keep all five in the primary comparison. Their head overlap is an explanatory observation, not a factorial isolation of L22H7's effect inside each larger group. Do not discard strong controls to restore a preferred conclusion.

The L22H7 controls also include L22H10, with effects +0.2978/+0.3347; L22H7 remains larger than every sampled single-head control. The L20H7 control sampler drew L20H11 twice, so its five draws contain four distinct head sets. The analysis preserves the frozen draw weights; five draws must not be described as five distinct heads or additional independent examples.

The ratio of mean L22H7 to mean three-head patch effects is 84.4% marked and 81.3% unmarked. This is neither an accuracy percentage nor evidence that this head accounts for that fraction of the model's retrieval ability.

## Answer Onset Is Not Complete Answer Transfer

No patch produces the full donor answer, including any control: **0/7,680 patch condition rows**. Those are repeated conditions on 160 identities, not 7,680 independent trials. The frozen extraction metric and the additional content diagnostic also find no complete donor numbers. Every full donor-minus-recipient likelihood margin remains negative; the largest is -4.0463. A zero count does not establish a zero population transfer rate.

A clean-donor answer in the corrupted recipient context would be *answer transfer*, not improved accuracy on that recipient's document.

For an answer of K tokens, the saved scores allow this decomposition:

```text
sequence_delta = K * delta_mean_logprob
first_delta = patched_first_logprob - baseline_first_logprob
remaining_tokens_delta = sequence_delta - first_delta
```

| Query-step patch | Sequence log-probability delta | First-token delta | Remaining five tokens, summed delta |
| --- | ---: | ---: | ---: |
| L22H7, marked | +6.2629 | +8.7858 | -2.5229 |
| L22H7, unmarked | +6.2297 | +8.4184 | -2.1887 |
| Address group, marked | +7.4248 | +10.4545 | -3.0297 |
| Address group, unmarked | +7.6615 | +10.2912 | -2.6297 |

The aggregate likelihood improvement comes from the first donor digit; the later donor digits become less likely in aggregate under teacher forcing. These later scores condition on the supplied donor-answer prefix, not on the freely generated sequence. The intervention itself is removed before those decoding steps.

The pilot highlighted one shared-first-digit example. Here there are 19 such identities and 141 whose first digits differ:

| L22H7 patch subgroup | n | Marked mean delta | Unmarked mean delta |
| --- | ---: | ---: | ---: |
| Different first digit | 141 | +1.1843 | +1.1774 |
| Same first digit | 19 | +0.0012 | +0.0061 |

The same-first-digit intervals include zero: [-0.0093, 0.0114] marked and [-0.0044, 0.0163] unmarked. This is consistent with the onset interpretation. It is not yet a controlled common-prefix experiment: the two subgroups contain different identities and answer pairs.

Among the 141 different-first-digit identities, the three-head patch changes the generated first token to the donor's in **129/141 marked (91.5%)** and **111/141 unmarked (78.7%)**, while complete donor generation stays zero. L22H7 alone does so in 19/141 and 8/141. The 19 shared-first-token cases must not be counted as successful first-token transfers merely because their first tokens already agree.

For example, clean `773974` and recipient `500835` produce `700835` after the address-group patch in both marker modes. Clean `336438` and recipient `109511` produce `309511`. Other outputs are shifted, shortened, or match the unrelated distractor, so they are not all simple one-digit substitutions. The complete changed-output ledger is saved with the analysis.

This establishes an effect of **patching these head outputs at this position**, not that a head intrinsically represents only one digit. Untouched document caches and subsequent decoding may matter. That is what the temporal follow-up should test.

## Ablation Separates Wording From Answer Selection

All ablations in this batch zero head outputs at the final prompt token only. They do not remove a head throughout document processing or later decoding.

| Ablated group | Marked exact | Marked correct-number content | Unmarked exact | Unmarked correct-number content |
| --- | ---: | ---: | ---: | ---: |
| None | 160/160 | 160/160 | 160/160 | 160/160 |
| L22H7 | 160/160 | 160/160 | 160/160 | 160/160 |
| L20H7 | 160/160 | 160/160 | 160/160 | 160/160 |
| Address group, 3 heads | 146/160 | 146/160 | 153/160 | 153/160 |
| Non-address core, 16 heads | 141/160 | 158/160 | 137/160 | 156/160 |

### The core mostly changes wording

Of the core's 19 marked exact-match failures, 17 still contain the correct number and two output only a person's name (`Bela`). Of its 23 unmarked failures, 19 still contain the correct number; three output a name and one outputs `ember12345`. These are observed output categories, not direct labels for failed internal retrieval or answer-type mechanisms.

For literal questions, correct-number content survives all 32 examples in both marker modes. Exact answers fall to 19/32 marked and 15/32 unmarked because outputs become sentences such as `The access code is 846940.` All outputs terminate at EOS, so this is not a token-budget artifact.

Almost the entire core-ablation sequence-score loss occurs at the first answer token: -0.2915 marked and -0.4391 unmarked, with remaining-token sums approximately zero. However, full candidate-margin shifts remain -0.9964/-1.4710. The group is therefore not proven to have a purely stylistic role.

The core's content loss is not clearly greater than its matched-control average: the focal-minus-control correct-content contrast is +0.50 percentage points [-1.50, 2.125] marked and -0.875 points [-3.50, 1.25] unmarked. Its larger strict-format loss must not be substituted for distinctive retrieval-content damage.

### The address group sometimes selects the distractor

All 14 marked and seven unmarked address-group ablation failures are in the alias template. Alias exact accuracy is 18/32 marked and 25/32 unmarked; all four other variants remain 32/32.

Seven marked failures and all seven unmarked failures exactly equal the explicitly unrelated distractor number. For example, the fact assigns NOVA key `336438`, the distractor note contains `970707`, and the ablated output is `970707`. The other seven marked failures are different or incomplete numbers. This distractor classification is a post-confirmation diagnostic grounded in the saved prompts, not a prespecified primary endpoint.

This is genuine answer-content failure, unlike merely adding a sentence around the correct answer. It also cannot be attributed to L22H7 alone: its individual query-step ablation changes no generated answers. Joint ablation is compatible with redundancy or interactions, but neither is isolated by this batch. A 16-head matched control containing L21H11 also produces distractor errors, so answer-selection behavior is not unique to our selected three-head group.

Finally, all focal ablated outputs still have positive *clean-versus-corruption-candidate* margins, even when their generated answer is wrong. The unrelated distractor was not the scored corruption candidate. A two-candidate preference score cannot establish open-ended correctness; follow-up scoring should explicitly include the unrelated distractor too.

## What The Publication Can And Cannot Claim

The historical atlas identified useful intervention targets. The pilots exposed an answer-start scoring issue and a mismatch between likelihood and generated content. The fresh 8K confirmation now replicates the narrower phenomena with frozen heads, more examples, and stronger controls:

- L22H7 and the address group shift donor likelihood beyond their matched controls, including without explicit needle markers.
- Query-step patching can strongly change answer onset without transferring a complete answer.
- Core ablation often breaks the requested output format while leaving the retrieved number in the output.
- The address group's joint ablation causes a template-concentrated answer-selection failure, including distractor selection.
- A binary classification of heads into answer donors and content-free support heads is unsupported. The core has a nonzero patch effect but is not unusually strong relative to its broader controls.

The evidence remains one model, five legacy templates, random Latin filler, one early answer position, and answer-replacement corruption. No new-template or natural-document generalization, long-distance specificity, decoding-time mechanism, complete circuit, or second-model replication has been established. Confirmation improves internal confidence in these observations; it does not by itself settle novelty or guarantee publication.

Keep the arXiv-first preparation and subsequent TMLR strategy, but revise the manuscript's claims before release. The useful research question is now how intervention position and measurement choice distinguish answer onset, answer selection, and response formatting. The general warning that metrics can disagree is not sufficient novelty by itself.

## Next Experiment

Implement a new, separately versioned temporal-patching runner. Do not change the historical or confirmation runner to silently redefine its intervention scope. The following is the proposed bounded design, not an already implemented or executed command:

| Element | Proposed choice |
| --- | --- |
| Model and interface | Same pinned Qwen revision and tokenizer; chat, unmarked, target 8K, early answer |
| Examples | 8 fresh identities per variant, 40 total; reserve seed base 11,100,000 with the existing variant offsets and explicit overlap checks |
| Corruption pairing | Two equal-length corruptions per identity: different first digit; same first digit but different second digit. Keep the clean fact, unrelated distractor, filler, and query matched |
| Focal patches | Frozen L22H7 and frozen three-head address group |
| Controls | Their existing five fixed layer/count-matched draws each; no head re-selection |
| Scopes | Final prompt token only; later decoding steps only; final prompt token plus later decoding steps |
| Identity checks | Clean and corrupt self-patches at each scope, including output token IDs and per-step hook counts |
| Main readouts | Full donor exact output, recipient exact output, unrelated distractor output, first differing digit and suffix behavior, per-token likelihoods, full margins, decoded text, EOS/budget termination |

This is 40 independent identities, 80 paired-corruption cases, and 44 condition rows per case: two baselines, six self-patch checks, and 12 patch groups across three scopes. The proposed budget is **3,520 condition rows**, with dependence retained across corruption types and scopes. It is a diagnostic budget, not a power calculation. Validate on a small disposable smoke split before running these reserved examples; freeze implementation details and hashes before collecting their outcomes.

The essential implementation rule is prefix alignment. At every free-generation step, the unpatched clean-context donor and the corrupted-context recipient must receive **the same prefix actually generated by the recipient**. Maintain separate caches and recompute the donor activation at the corresponding position. Never feed the donor's known correct answer and call the resulting transfer free generation. Teacher-forced candidate branches are separate measurements with their own matched prefixes and isolated caches.

Later-only patching cannot repair an already wrong first digit. The same-first-digit condition makes later-only full-answer recovery possible in principle; evaluate suffix effects separately when first digits differ. Balanced pairing removes the major limitation of interpreting the current 19-versus-141 subgroup comparison.

If continued patching transfers full answers above controls, it supports repeated participation during decoding rather than a one-time complete answer transplant. If it does not, retain the limited onset claim and investigate other positions or pathways before proposing a larger mechanism. Either result constrains the interpretation; neither licenses a complete-circuit claim.

After this diagnostic, run the previously planned answer-start/local-task test: raw versus explicitly separated raw versus chat, and distant versus nearby facts in matched-length contexts. A short/local copy condition can be additional, but should not be conflated with a pure distance comparison. This tests whether the core's apparent support role is retrieval-specific or mostly answer initiation/format control. Keep unrelated-distractor scoring and strict/content metrics separate. The alias failures also motivate the later competing-fact/query-selection task.

Only then spend on new prose/templates and a targeted second-model replication. Do not start broad 16K sweeps, re-select heads on these confirmation outcomes, or submit the old mechanism claim unchanged. No new GPU experiment or publication submission has been performed in this analysis pass.

## Reproduction

```bash
.venv/bin/python scripts/diagnose_publication_confirmation.py
.venv/bin/python -m unittest discover -s tests -v
```

The new analysis reuses the existing integrity checks and paired bootstrap implementation. `artifacts_publication/confirmation_8192_chat_diagnostics.json` contains input hashes, 204 condition/variant summaries, first-token subgroups, each fixed control's composition and effect, content contrasts, the chat pilot comparison, and all 646 changed outputs across focal and control conditions. The original five confirmation files and both pilot runs are unchanged. All 26 local tests pass; model-level tests use a tiny random CPU model, not the pretrained GPU experiment.
