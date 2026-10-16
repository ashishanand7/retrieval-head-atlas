# Research Manuscript Outline

Evidence snapshot: 16 October 2026, analysis commit `38cc7f966678b8c62423e71acfd1c33f40b4d9bb`; latest experiment outputs in `64b6380`. This is the content and evidence plan for a separate research manuscript, not an expansion of the capstone report.

## 1. Paper Identity And Argument

**Working title:** From Answer Likelihood to Answer Selection: Auditing Retrieval-Head Interventions in Qwen2.5

**Research question:** When does a causal change in an answer-likelihood score correspond to a change in the complete answer a model generates?

**Central claim:** In controlled six-digit factual lookup with Qwen2.5-1.5B-Instruct, a frozen three-head intervention can redirect complete answers between facts in a document. Comparable likelihood gains do not produce complete transfer in earlier answer-replacement tasks, and boundary, timing, and ablation controls show why score changes alone do not establish a head's computational role.

**Lead finding:** On 40 held-out multi-fact identities, the three-head combined patch switches the far query-selection recipient to the target answer in 32/40 cases, versus 0/40 for the baseline and each of five fixed matched controls. The paired control-adjusted effect is +80.0 percentage points, with a 95% pointwise interval of [67.5, 90.0]. The document is unchanged between donor and recipient; only the question differs. This is causal answer redirection under a specified intervention, not improved accuracy on the recipient's own question.

**Reader takeaway:** Evaluate retrieval-head interventions with complete generated outputs as well as token scores, aligned temporal interventions, multiple corruption designs, fixed matched controls, and a donor-reproduction calibration. Our contribution is the concrete, audited case study and its positive and negative behavioral results, not the general discovery that patching metrics can disagree.

Aim for approximately 8-10 main-text pages, excluding references and appendices, with four principal figures and two tables. This is an editorial budget, not a venue rule. Tell a scientific argument, not a semester-by-semester experiment diary. Introduce the historical atlas briefly as the source of the frozen intervention targets.

## 2. Claims And Evidence

All likelihood effects below are natural-log probability changes per target token. Exact-transfer effects are percentage points, not relative percentage improvements. Full-run multi-fact cells repeat the same 40 identities. Frozen primary/secondary contrasts and later diagnostics must retain their different evidential status.

| Claim | Main evidence | Status and boundary | Placement |
| --- | --- | --- | --- |
| **C1. The frozen head set can redirect complete answers between available facts.** | Multi-fact far query selection: three heads produce 32/40 targets; every matched control produces 0/40. Far value swap produces 29/40 targets, of which 28 are newly induced. Near counts are 38/40 and 35/40 respectively. | Far query selection, combined scope, focal-minus-control is the frozen principal comparison. Other settings are secondary. Does not identify a pure selection algorithm, establish minimality, or improve recipient-task correctness. | Results 4.1; F1; T2. |
| **C2. The size of a likelihood gain does not by itself establish complete answer transfer.** | Far three-head combined patches: earlier specificity, different-first stratum, +1.9727 with 0/40 complete targets; new value swap +1.9595 with 29/40; new query selection +2.0541 with 32/40. Specificity's broad positive control nevertheless reproduces every donor. | Cross-study comparison is descriptive, not a paired estimate. Prose, fact inventory, and corruption change together. The earlier null is conditional, not evidence that these heads can never transfer answers. | Results 4.2; F2. |
| **C3. Temporal interventions affect more than answer onset, but continuing a patch is not a universal route to complete transfer.** | Earlier temporal study: three-head different-first score gains are +1.4523 prompt-only, +0.3970 decode-only, +2.0502 combined, with 0/40 complete targets in each. Shared-first decode-only changes later-token scores. In multi-fact tasks, combined patching repairs 2/0/7/4 outputs relative to prompt-only for far swap/far query/near swap/near query. | Three scopes and two corruption strata were frozen in the temporal study. Multi-fact output-transition counts are post-run diagnostics; far query selection has no additional decoding benefit. No claim of an exclusively first-token role or a complete decoding circuit. | Results 4.3; F3. |
| **C4. Boundary and replacement choices change what ablation damage means.** | Specificity: core zero-ablation raw score penalty -1.1981 becomes -0.0045 with one supplied space; the correct first number remains 40/40. Multi-fact near core zero leaves 32/40 exact outputs but 40/40 correct-number content and candidate rank; core resampling preserves all 40 exact answers. | The boundary changes an input token and the intervened activation, not only a scoring formula. First-number analysis is post-run in specificity. Resampling can inject incompatible information. This does not prove a content-free core or strict necessity of the three-head group. | Results 4.4; F4. |

The primary C1 likelihood endpoint also favors the focal patch: +2.2279 [2.0680, 2.3714] relative to the five-control mean. Report it alongside the behavioral endpoint, not instead of it. The raw focal gain is +2.0541; these are different quantities.

Far query-selection donors are correct on 37/40 identities and recipients on 39/40 for their own questions. Keep all 40 in the main result; the post-run subset with both baselines correct gives 31/36 transfers and belongs in sensitivity analysis, not in a replacement headline.

### Supporting Results And Counterexamples

- **Both multi-fact corruptions work.** Query-minus-swap control-adjusted exact effects are +10.0 pp [-7.5, 27.5] far and +7.5 pp [-5.0, 20.0] near. Neither establishes a complete-answer advantage or equivalence. Their corresponding score contrasts are positive, but partly reflect different control effects.
- **Effects are not confined to distant facts.** New combined patches are stronger near the question: far-minus-near exact effects are -17.5 pp [-32.5, -2.5] for value swap and -15.0 pp [-30.0, -2.5] for query selection. Earlier specificity also finds score effects in near and short tasks. Position moves with distance; short tasks additionally remove context.
- **Calibration is successful, not a discovered mechanism.** Full-residual replacement reproduces actual donor token sequences and termination in 320/320 specificity cases and 160/160 multi-fact cases, with zero recorded logit error. These are repeated cases on 40 identities in each study. Multi-fact target correctness is 154/160 because six cases reproduce incorrect donor outputs.
- **A strong group effect is not minimality.** L22H7 alone yields 8/40 far query-selection targets versus 32/40 for the three-head group. This comparison does not isolate synergy, redundancy, or the contributions of the other two heads.
- **Important failures stay visible.** Seven of the eight far query-selection failures involve the third fact, as do all three far clean-baseline errors. This is a post-run association across fixed identities, not a randomized explanation. Mention it in the main limitations and show the full slot/family breakdown in the appendix.
- **Controls remain fixed.** Some matched controls change likelihood and some overlap active heads. Retain them all. Zero newly induced targets among the multi-fact controls is an observed result, not proof of zero population probability; repeated control rows do not increase N.

## 3. Experiment And Source Map

The following study labels are manuscript shorthand, not new experiments. Link each result to its frozen run and analysis; do not pool separate studies into a single effectiveness estimate.

| ID | Study and independent confirmation units | Manipulation and purpose | Primary source |
| --- | --- | --- | --- |
| E0 | Historical atlas, exploratory | Select candidate heads/groups; context-length, position, ablation and patching sweeps. Discovery provenance only in the main text. | [Publication readiness audit](../docs/publication_readiness_audit_2026-09-20.md), [historical numerical tables](../artifacts_phase2/report_assets/tables/table_main_results.md). |
| E1 | Raw/chat pilots, 10 paired identities shared across interfaces | Expose answer-boundary and score/output disagreement. Engineering and hypothesis development, not extra confirmation data. | [Pilot analysis](../docs/publication_pilot_analysis_2026-10-14.md), [saved comparison](../artifacts_publication/pilot_comparison.json). |
| E2 | Held-out chat confirmation, N=160, five template variants | Marked/unmarked prompts, answer replacement, prompt-step patches/zero ablation, five controls per group. | [Protocol](../docs/publication_confirmation_protocol.md), [run](../artifacts_publication/confirmation_8192_chat/), [analysis](../docs/publication_confirmation_analysis_2026-10-15.md). |
| E3 | Temporal confirmation, N=40, five variants | Paired different-first/shared-first corruptions, three scopes, candidate scores and free generation. | [Protocol](../docs/publication_temporal_protocol.md), [run](../artifacts_publication/temporal_confirmation_8192_chat/), [analysis](../docs/publication_temporal_analysis_2026-10-15.md). |
| E4 | Specificity/boundary confirmation, N=40, five variants | Far/near/short chat, raw/raw-space boundary, zero/resampling, positive donor calibration. Not every manipulation is fully crossed. | [Protocol](../docs/publication_specificity_protocol.md), [run](../artifacts_publication/specificity_confirmation_8192/), [analysis](../docs/publication_specificity_analysis_2026-10-16.md). |
| E5 | Multi-fact confirmation, N=40, four new prose families | Four candidate facts, paired query selection/value swap, far/near positions, prompt/combined patches, zero/resampling and positive control. | [Protocol](../docs/publication_multifact_protocol.md), [run](../artifacts_publication/multifact_confirmation_8192/), [analysis](../docs/publication_multifact_analysis_2026-10-16.md). |

E3/E4 engineering smokes have five identities each; E5 smoke has eight. Keep them separate. Reference prompts and repeated marker, corruption, distance, scope, and head-control conditions are not independent examples. E2-E5 study identities are disjoint; every study still uses the same model. The latest audit checked the new study/reference schedules against 345 earlier study and reference identities, not 345 independent confirmations of the current effect.

E5 actual lengths are 8,153-8,192 tokens, with source-end-to-question gaps of 7,151-7,332 tokens far and 6-109 near across its four facts. Older studies labeled 8K include chat/template overhead and can exceed 8,192 tokens; report their actual ranges separately rather than claiming all prompts have exactly the same length.

## 4. Section Outline

### Abstract

Write last, approximately 180-220 words. Plan five moves:

1. Motivate the gap between identifying intervention-sensitive heads and demonstrating a complete generated answer change.
2. State the single-model, frozen-head, controlled factual-lookup design and its behavioral, temporal, boundary, and positive controls.
3. Lead with 32/40 far query-selection targets and the matched-control comparison. Include the independent N and make clear that the intervention redirects the answer.
4. Contrast earlier likelihood gains without complete transfer and the answer-boundary effect without listing every experiment.
5. Conclude with conditional behavioral efficacy and the limits of score-based role inference. Do not claim a universal circuit, a new general patching method, or isolated causation by answer availability.

### 1. Introduction

**Question:** What does an important retrieval head actually allow us to predict about model behavior?

Open with the distinction between raising the likelihood of an answer and making the model emit it. Use one short multi-fact example to make the intervention tangible before introducing head indices. Explain why both positive transfers and controlled failures matter for a mechanistic account.

Summarize the historical Retrieval Head Atlas in one paragraph: it supplied exploratory intervention rankings and the fixed candidate groups, not a previously established functional decomposition. Avoid a capstone recap or a general tutorial on transformers.

Proposed contributions:

1. A held-out demonstration of complete question-conditioned answer switching by a frozen three-head patch in synthetic multi-fact lookup, with paired value-swap, distance, and matched-head controls.
2. A comparative empirical account of when comparable score effects do and do not translate to generated answers, including temporal and answer-boundary counterexamples.
3. An auditable evaluation package with shared-prefix donor execution, dense positive calibration, separate scoring/generation endpoints, immutable traces, and identity-level inference. This is reproducibility infrastructure, not an invented causal-identification method.

End with one sentence delimiting the model, task family, and unresolved mechanism. Use the F1 result preview; save detailed experimental history for the appendix.

### 2. Related Work

Organize by the scientific distinction rather than author-by-author summaries: retrieval-head localization; temporally varying retrieval behavior; activation-patching methodology and inference; long-context task/position sensitivity.

Primary-source anchors checked for this outline:

| Anchor | Overlap to acknowledge | Distinction the manuscript must substantiate |
| --- | --- | --- |
| [Wu et al., Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574) | Sparse heads and their causal relevance to contextual retrieval are established subjects of study. | Our contribution cannot be merely locating another retrieval head. Compare intervention sites, head selection, task design, and complete-answer endpoints. |
| [Lin et al., Retrieval Heads are Dynamic](https://aclanthology.org/2026.acl-long.715/) | Retrieval behavior across generation timesteps is already studied. | Our timing study holds the selected heads fixed and contrasts donor-score changes with complete generated transfer; it does not discover time-varying head sets. |
| [Zhang and Nanda, Towards Best Practices of Activation Patching in Language Models](https://arxiv.org/abs/2309.16042) | Metrics and corruption methods can change interpretability conclusions. | Claim the specific retrieval-task evidence and controlled behavioral contrast, not metric sensitivity itself as a new idea. |
| [Heimersheim and Nanda, How to Use and Interpret Activation Patching](https://arxiv.org/abs/2404.15255) | Patching evidence has limits for claims about circuits and computational roles. | Situate our positive/null results and calibration within those limits; do not equate local behavioral efficacy with circuit recovery. |

Before writing novelty language, complete a methods-level comparison with these papers and the retrieval/support, synthetic-context, and semantic-retrieval work listed in the [publication strategy](../docs/publication_strategy_2026-09-20.md). Add appropriate model, transformer, and position-sensitivity references from the [existing bibliography](../paper/references.bib), verifying metadata and relevance rather than copying the bibliography wholesale. The present anchors are not an exhaustive literature review. No "first" claim is authorized by this outline.

### 3. Methods

Drafted in [methods.md](methods.md) on 17 October 2026, with [implementation traceability](methods_traceability.md). The subsections below remain the planning checklist; the draft expands them into the implemented study designs and equations.

**3.1 Model and frozen head selection.** Qwen/Qwen2.5-1.5B-Instruct at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; pinned tokenizer/chat template; bfloat16 and SDPA. Distinguish the query/output head slices from grouped key/value heads. Introduce H = {L22H7, L21H11, L22H10}, the single-head L22H7 comparison, and the exploratory 16-head group C. Indices are zero-based. Use "three-head group" and "16-head group" in publication prose; place all indices and discovery selection details in the appendix. The [frozen group configuration](../configs/publication_confirmation.json) is authoritative.

**3.2 Paired prompts and tasks.** Define donor context, recipient context, target code, and recipient-correct code. Explain E5 first: four plausible person-code facts, readable synthetic background, no markers, distinct first digits, balanced fact slots, and held-out confirmation wording across dispatch/archive/workshop/reservation families. Query selection changes only the requested entity; value swap exchanges two code spans but preserves the question and all four candidates. Then introduce E2-E4 answer replacement, markers, shared-first corruption, and the boundary/near/short controls. Use T1 for the compact design inventory.

**3.3 Intervention site and temporal scope.** Replace selected head slices at the input of the attention output projection (`o_proj`), not attention weights or separate Q/K/V pathways. Document prefill remains unpatched. Prompt-only means the last prompt position; decode-only skips the first answer prediction; combined acts at both stages, including EOS prediction. Decode-only belongs to E3, not a missing E5 arm. At each intervened step, the unpatched donor uses the same prefix actually consumed by the recipient branch. Teacher-forced and greedy branches have independent caches.

**3.4 Controls and replacements.** Five frozen head-count/layer-matched controls per focal group; self-patches; task-matched one-step resampling for the clean-prompt ablation panel; full-residual donor replacement immediately before final normalization at every prediction for positive calibration. The dense control must reproduce actual donor logits, token sequence, and stopping, including donor errors. It is not an accuracy ceiling or a sparse circuit discovery. Zero/resampling interventions in E4/E5 are prompt-only, not whole-run head removal.

**3.5 Endpoints and estimands.** Define a known target code independently of the donor's actual output. Exact target answer permits surrounding whitespace; verbatim format does not. Correct-number content permits repeats of the same target but no competing six-digit code. Keep first-number, four-candidate rank, candidate margins, and output termination as separate diagnostics. E5 scores all four candidates; E3/E4 score three. Teacher forcing excludes EOS and supplies later answer prefixes; greedy output does not receive the target answer.

Minimum equations for the Methods draft:

```text
Head replacement at active positions:
    z_recipient[layer, head, t] <- z_donor[layer, head, t]
    for (layer, head) in H and t in the chosen intervention scope.

Mean target-token score (K = 6 for these tasks):
    L_i(I) = (1/K) * sum_t log p_I(y*_it | x_recipient_i, y*_i,<t)
    delta L_i(I) = L_i(I) - L_i(baseline)

Complete target output:
    E_i(I) = 1[strip(decoded greedy output_i(I)) = target code_i]

Matched-control effect for endpoint m, with R_j the five fixed groups:
    theta_m = mean_i[delta m_i(H) - (1/5) * sum_j delta m_i(R_j)]
```

The baseline in the last equation is the recipient for donor patching and the clean prompt for zero/resampling. Express `100 * theta_E` in percentage points. Count newly induced targets separately as `E_i(H)=1` with `E_i(baseline)=0`; this matters for the one far value-swap baseline coincidence. These equations specify measured effects, not a mediation model or a recovered causal graph.

**3.6 Statistical and procedural commitments.** Name the frozen E5 principal comparison and its two endpoints. Average controls within identity, preserve paired conditions, and bootstrap identities within template family with 4,000 draws. State 95% pointwise intervals, no multiplicity adjustment, and conditioning on fixed control draws. These intervals do not quantify uncertainty over new models, unseen template families, or new control draws. Retain all baseline failures in primary analyses. Describe post-run token decompositions, output transitions, slot breakdowns, and baseline-success sensitivities as diagnostics. Protocols were committed before the relevant outcomes; do not call this formal external preregistration. Later study designs were informed by earlier results.

**3.7 Reproducibility.** State the SageMaker execution environment from manifests, actual prompt ranges, generation budgets (24 tokens in E2/E3, 64 in E4/E5), hash/audit checks, and zero-error self/positive controls. The reported output-only interpretation remains censored for budget-limited continuations. Put software versions, exact source revisions, all commands, and the complete integrity ledger in the appendix.

### 4. Results

**4.1 Frozen heads switch complete answers in multi-fact lookup (C1; F1, T2).** Lead with the query-only manipulation and 32/40 primary result. Show recipient baselines and all controls, then value swap and near views. Explain the 29 total versus 28 new far-swap targets. Report single-head results as a secondary scale comparison, not a minimal-circuit test. Include the uncertain query-minus-swap behavioral contrast and stronger near effects without converting either into a pure selection/transport interpretation.

**4.2 Similar score gains can accompany different generated outcomes (C2; F2).** Place earlier answer replacement beside the two new tasks using raw per-token gains and exact-target counts. State the task differences at the comparison, not only in Limitations. Briefly connect the N=160 unmarked confirmation (+1.2769 prompt-step gain, 0/160 complete targets) to its independent temporal and specificity follow-ups. The new result qualifies the old sparse-transfer null; it does not erase it. Full-residual calibration rules out a pipeline simply incapable of donor reproduction, not every possible sparse-patch design problem.

**4.3 Continuing patches changes later behavior conditionally (C3; F3).** Show the E3 scope comparison, including shared-first-digit corruption, so later effects are not confused with a changed first digit. Explain why increasing a very small probability does not force a greedy decision. In E5, show the extra complete answers from combined versus prompt-only patching and one hybrid-continuation repair; far query selection remains unchanged. Keep prefix-match examples subordinate to full-answer results. No document-prefill or necessity claim follows from this timing experiment.

**4.4 Boundary, formatting, and ablation choice alter the interpretation (C4; F4).** Start with the paired raw/raw-space core score contrast: control-adjusted repair +1.1491 [1.1083, 1.1920], while all 40 initial numbers remain correct. Distinguish initial number, exclusive number content, strict answer, and censored explanation. Show that the issue persists beyond raw prompts: new-prose near core zero reduces exact output to 32/40 while content and candidate rank remain 40/40. Contrast zero with resampling for both groups; near three-head resampling leaves 8/40 exact outputs but ranks the target first in 27/40. Injection of competing reference activity is not simple deletion, so greater resampling damage is not proof of necessity.

### 5. Discussion

Synthesize the positive and negative evidence as **conditional behavioral efficacy of a specified intervention**. A frozen head set can causally steer answer choice in these tasks without being a context-independent representation of the complete answer or an identified minimal retrieval circuit.

Discuss answer availability as a plausible explanation for the earlier/new task contrast: the old recipient lacks the target code, whereas all four codes remain present in the new design. Explicitly separate that hypothesis from the controlled result that query-selection donor and recipient documents are identical. Other pathways, relative positions, and prefix-dependent dynamics remain possible contributors.

Offer a compact reporting recommendation grounded in the case: show free generation with score changes; expose first/later token contributions; preserve recipient-prefix alignment; include successful broad calibration and fixed controls; state corruption, boundary, scope, and baseline performance. Attribute the general methodological principles to prior work.

### 6. Limitations And Reproducibility Statement

One 1.5B instruction-tuned model, deterministic generation, six-digit single-token-per-digit answers, a small number of synthetic families, and N=40 for the latest task constrain generalization. New prose is not a natural-document benchmark or multi-hop reasoning evaluation. The selected set is frozen but was discovered exploratorily; discovery ranks are not untouched confirmation evidence.

Report the third-fact failure concentration, imperfect donors, overlapping controls, pointwise uncertainty, and the distinction between paired within-study and unpaired cross-study comparisons. Moving a fact cluster changes both distance and absolute position. The positive control acts near the output and does not isolate how the model retrieved the information. There is no separate Q/K-versus-V, path-patching, or minimality experiment.

Describe artifact access, extraction requirements for the historical compressed specificity trace, and runnable analyses without making untested portability claims. Confirm human authorship, contributions, affiliations, funding/compute acknowledgments, and any required AI-assistance disclosure before release. Keep administrative account setup out of the research narrative.

### 7. Conclusion

Return to the concrete result: complete switching among present facts under frozen sparse interventions, contrasted with likelihood gains without complete transfer in other controlled settings. Conclude that behavioral claims about retrieval heads require task- and endpoint-specific evidence. Do not end by upgrading a bounded positive result into general retrieval, accuracy, or circuit guarantees.

## 5. Main Figure Plan

These are specifications for manuscript-specific figures, not a claim that the final layouts already exist. The three current diagnostic figures are reusable source material. Keep their data and historical captions intact; generate the paper's layouts separately in this directory during the figure pass.

### F1. A Frozen Three-Head Patch Redirects The Answer

**Purpose:** Explain the task and establish C1 before introducing the earlier nulls.

- **A, paired-task schematic:** Four named facts, unchanged query versus swapped values, and unchanged document versus changed query. Use the recorded `dispatch:15100000` example: Rose `289307`, Emma `373000`, Nina `733766`, Maya `453485`. For query selection, the recipient asks for Emma and changes from `373000` to `289307` under the Rose-question donor patch. Label any omitted background as schematic.
- **B, complete target outputs:** Plot baseline, L22H7, three-head combined, and the fixed-control mean for far/near query selection and value swap. Use counts out of 40 and display the five draws individually where legible. Mark the far-swap pre-existing target distinctly so 29 outputs are not presented as 29 induced switches.
- **C, primary effect:** A compact paired contrast with +80.0 pp [67.5, 90.0] for far query selection; mark its prespecified status. Put the other paired contrasts in T2 rather than crowding the lead figure.

**Source:** E5 [paired analysis](../artifacts_publication/multifact_confirmation_8192/paired_analysis.json), [diagnostics](../artifacts_publication/multifact_confirmation_8192_diagnostics.json), and examples/rows. Reuse [multi-fact figure](../artifacts_publication/figures/multifact_transfer.pdf) A/B data via [its diagnostic script](../scripts/diagnose_publication_multifact.py). The schematic and final composition are new work for the figure pass.

**Caption must say:** selected-head intervention, known target rather than necessarily actual donor output, repeated N=40 identities, fixed controls, pointwise CI, and answer redirection rather than recipient-task accuracy gain.

### F2. Comparable Likelihood Gains Do Not Guarantee Complete Transfer

**Purpose:** Establish C2 without implying an isolated answer-availability experiment.

- **A/B, aligned small multiples:** E4 far answer replacement (different-first stratum), E5 far value swap, E5 far query selection; three-head combined raw score gains +1.9727/+1.9595/+2.0541 beside complete-target counts 0/29/32 out of 40. Use separate axes for score and output count, not a dual-axis plot or pooled correlation. Annotate 28 newly induced in value swap.
- **C, calibration:** In the earlier E4 far chat different-first setting, sparse transfer is 0/40 while dense donor reproduction is 40/40. Separately annotate the E5 audit as 160/160 actual donor reproductions over repeated cases, including six incorrect targets. Dense calibration is not a sparse baseline or an accuracy ceiling.

**Source:** E4/E5 [cross-run diagnostics](../artifacts_publication/multifact_confirmation_8192_diagnostics.json) and [specificity diagnostics](../artifacts_publication/specificity_confirmation_8192_diagnostics.json); reuse [specificity figure](../artifacts_publication/figures/specificity_controls.pdf) B with explicit stratum labels. The cross-run composition is new; add no between-study significance bracket.

**Caption must say:** independent batches, different templates/fact inventory/corruption, descriptive comparison, separate N=40 per study, and donor reproduction distinct from correctness. These three aggregate points do not estimate a general score-behavior correlation.

### F3. Timing Changes Scores And Some Complete Continuations

**Purpose:** Establish C3 and explain the shared-prefix intervention.

- **A, timeline:** Last prompt position, subsequent answer predictions, and EOS; show the three E3 scopes. Label later donor input as the recipient branch's actual prefix, never the desired answer. No patch during document prefill.
- **B, earlier-task scopes:** E3 paired different-first/shared-first score effects for the three-head group with pointwise intervals, next to the zero complete-transfer result. Use control-adjusted effects consistently in this panel and label them as such. Full L22H7 and prefix-length results can move to the appendix.
- **C, new-task continuations:** Prompt-only to combined target counts 27 to 29, 32 to 32, 28 to 35, 34 to 38 out of 40. Annotate the post-run paired repairs (2/0/7/4, no losses) and the recorded near example `826775` to `826763` for target `826763`.

**Source:** E3 [paired analysis](../artifacts_publication/temporal_confirmation_8192_chat/paired_analysis.json), [diagnostics](../artifacts_publication/temporal_confirmation_8192_chat_diagnostics.json), and [temporal script](../scripts/diagnose_publication_temporal.py); E5 scope transitions/repair ledger. The [existing temporal figure](../artifacts_publication/figures/temporal_scope_comparison.pdf) uses raw score changes; do not relabel those numbers as control-adjusted.

**Caption must say:** E3 and E5 are different studies; counts are paired within each; continuing the patch does not always help. Prefix-match and scope-repair ledgers are diagnostics, not replacement primary endpoints.

### F4. Scores, Number Content, And Output Format Are Different Readouts

**Purpose:** Establish C4 with one boundary manipulation and one clean-prompt intervention comparison.

- **A, answer boundary:** E4 raw/raw-space/chat core zero-ablation control-adjusted score effects, including the paired raw-space-minus-raw interval. Annotate correct first number 40/40 in raw and raw-space, but strict exact output 0/40 in both. A space repairs the scoring boundary, not raw exact accuracy.
- **B, behavioral readouts:** E5 near baseline, core zero, core resample, three-head zero, three-head resample; show exact target, exclusive correct-number content, and strict four-candidate rank separately. Include the two censored focal resampling outputs in the relevant counts and disclose their budget stops.

**Source:** E4 [specificity script](../scripts/diagnose_publication_specificity.py) and [figure](../artifacts_publication/figures/specificity_controls.pdf) C; E5 [multi-fact figure](../artifacts_publication/figures/multifact_transfer.pdf) D and diagnostics. Recompose as a two-panel manuscript figure; do not mix ablation on clean prompts with donor patches on corrupt prompts in one unlabeled scale.

**Caption must say:** prompt-only zero/resampling, clean recipients, separate E4/E5 identity batches, first-number diagnostics versus exclusive content, and teacher-forced candidate rank versus free generation. Label categories with dots/bars rather than implying a continuous trajectory across intervention types.

### Shared Figure Rules

Use deterministic plotting scripts and editable vector schematics; save vector PDF plus PNG preview. Use consistent head-group and scope legends, legible text at final column width, count denominators, units, and confidence-interval definitions. Do not reuse the capstone's stronger functional-role diagrams unchanged. No image-generated data plots, decorative neural-network illustrations, or arrows asserting an untested content/support pathway are needed.

## 6. Tables And Supplement

**T1, experimental design:** A compact E2-E5 table with independent N, template families, actual token ranges, corruption, temporal scopes, generation budget, and calibration availability. Put E0/E1 and smoke details in the appendix. State which contrasts are paired and that the task grid is not fully crossed. Use manifests/examples and the source map above.

**T2, multi-fact paired effects:** Report the principal far query-selection contrast first, then far value swap, query-minus-swap far/near, and far-minus-near for both corruptions. Columns: contrast, target-token score estimate/CI, exact-target percentage-point estimate/CI, and primary/secondary status. Pull unrounded values from E5 `paired_analysis.json`; keep raw target counts in F1. Do not use "accuracy improvement" as the column label.

| Appendix | Material and purpose |
| --- | --- |
| A. Discovery provenance | Historical atlas selection procedure, every frozen focal head, exploratory ranking/length/position sweeps needed to explain selection. No promotion of discovery data to confirmation. |
| B. Task and intervention details | Complete generators, prompt examples, code inventory, aliases, source spans, boundary tokenization, smoke/confirmation wording and seeds, shared-prefix cache/branch procedure. |
| C. Complete effects and controls | All fixed control head sets and overlaps, L22H7 comparisons, marker conditions, same-first strata, paired contrast tables, zero/resampling panels, sensitivity to baseline success. |
| D. Output and failure analysis | All baseline errors and changed outputs, family/slot breakdown, third-fact concentration, partial/hybrid answers, wrong-donor reproduction, censored continuations. State which analyses are post-run. |
| E. Reproducibility and calibration | Source/model/tokenizer/environment hashes, manifests, positive/self-control checks, exact regeneration commands, compressed-trace handling, test scope, measured runtimes and memory. No utilization claims from peak memory. |

Historical sweeps demonstrate how the targets were selected; they need not all become main-paper plots. The earlier N=160 confirmation, the new positive result, the dense calibration, and the limitations remain visible in the main text even when their exhaustive tables move to the supplement.

## 7. Claim Limits And Remaining Work

Do not claim that these are the only retrieval heads, a minimal/necessary/sufficient complete circuit, pure address-versus-content modules, or a method that improves general model accuracy. Do not claim that absent-versus-present answers cause the between-study difference, that query selection is definitively easier to steer than value swap, or that a null interval proves equivalence. Do not generalize from synthetic numerical lookup to natural-document reasoning or other models without evidence.

### Writing And Release Sequence

1. **Methods drafted; Results next.** Follow Sections 3 and 4 above, preserve C1-C4 and F1-F4 mappings, and link numerical claims to the frozen analyses. No further GPU run is required to start Results.
2. **Make manuscript-specific figures and tables.** Reuse audited data and plotting logic, add the precise task/timing schematics, and verify every caption against its endpoint and denominator. Keep diagnostic assets and both university reports unchanged.
3. **Complete the related-work comparison; then write Introduction, Discussion, Conclusion, and Abstract.** Test the proposed contribution against the closest work before making novelty language stronger. Do not let a title imply broader scope than the results.
4. **Run one targeted second-model-family study alongside drafting if pursuing the stronger journal version.** It needs separate head discovery, a feasibility/hook check, and held-out confirmation, not copied head indices. Model choice, fit, protocol, and job count remain to be specified; no result is assumed or blank results section inserted.
5. **Only add a matched answer-present/absent experiment if claiming that availability causes the observed difference.** Otherwise retain availability as a hypothesis. This is an optional stronger claim, not an indefinite barrier to the single-model paper.
6. **Freeze the content before formatting and release.** Review numerical/citation traceability, author information and disclosures, assemble a reproducible research package, then prepare the preprint/review layouts and recheck current submission requirements. Upload and account/profile changes are separate actions, not consequences of writing this outline.

The current evidence supports drafting a focused single-model manuscript now. A stronger cross-model mechanism claim, exhaustive novelty assessment, peer-review acceptance, and public indexing are not established by the completed experiments.
