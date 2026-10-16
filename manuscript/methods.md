# 3. Methods

We evaluate whether interventions on a fixed set of attention-head outputs change answer likelihood, generated answer content, or both. Four studies separate marker dependence, intervention timing, source proximity and answer boundary, and selection among multiple facts. Each uses paired prompts and within-identity intervention contrasts. The studies were conducted sequentially, with later designs informed by earlier findings; comparisons between studies are therefore descriptive, not isolated estimates of a single design change.

## 3.1 Model And Frozen Head Selection

We use the instruction-tuned **Qwen2.5-1.5B-Instruct** model without fine-tuning or weight updates ([Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)). All analyzed runs resolved to checkpoint `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` and the same tokenizer/chat-template fingerprint. The initial marker-confirmation runner recorded the resolved checkpoint; the subsequent runners explicitly requested that revision and checked it before evaluation. The model was evaluated in bfloat16 on an NVIDIA A10G using scaled dot-product attention, with float32 log-softmax for the reported token scores. Models were in evaluation mode and generation was greedy.

The model has 28 transformer layers and 12 attention-output heads per layer. We index layers and heads from zero. The principal group is

\[
H=\{(22,7),(21,11),(22,10)\}.
\]

These indices refer to head slices before the attention output projection, not to separate key/value heads. We also test L22H7 alone and a disjoint 16-head group, denoted \(C\). The marker-confirmation study additionally includes L20H7 as a single-head comparator. Group \(C\) is the complement of \(H\) within a previously selected 19-head set; its full membership is retained with the experiment configuration.

The groups originated in an exploratory study on 40 identities spanning five synthetic retrieval templates. Query-step ablation and answer-span attention were used to define candidate groups, followed by group and single-head patching. That exploratory work reused identities across selection and follow-up measurements. We therefore treat it as discovery, not as independent confirmation. All focal indices were frozen before the publication pilots and retained throughout the studies reported here. The historical implementation labels do not establish distinct computational roles for the groups.

## 3.2 Tasks And Paired Prompts

For identity \(i\), let \(x_i^d\) be the donor prompt, \(x_i^r\) the recipient prompt, \(y_i^*\) the known answer to the donor task, and \(y_i^r\ne y_i^*\) the answer appropriate to the recipient task. Both prompts describe valid tasks. A patch that makes the recipient emit \(y_i^*\) demonstrates answer redirection, not improved correctness on the recipient's own task. The known target \(y_i^*\) is also distinct from what the unmodified donor actually generates, which can be incorrect.

**Table 1. Confirmation-study design.** A paired case specifies an identity and its marker, view, or corruption condition; multiple interventions are then evaluated on that case. Neither cases nor intervention rows increase the independent identity count. Candidate counts refer to teacher-forced scoring, not a restriction on free generation.

| Study | Independent identities | Paired cases | Realized prompt tokens | Scored candidates | Generation budget |
| --- | ---: | ---: | --- | ---: | ---: |
| Marker confirmation | 160; 32 per legacy template | 320 | Marked: 8,220-8,222; unmarked: 8,208-8,211 | 2 | 24 |
| Temporal patching | 40; 8 per legacy template | 80 | 8,208-8,211 | 3 | 24 |
| Specificity and boundary | 40; 8 per legacy template | 320 | Long chat: 8,208-8,210; short chat: 101-119; raw views: 8,179-8,182 | 3 | 64 |
| Multi-fact selection | 40; 10 per new prose family | 160 | 8,153-8,192 | 4 | 64 |

### Multi-Fact Selection

Each document contains four contiguous person-code facts from one of four families: parcel dispatch, archive collections, workshop service records, or room reservations. Each fact assigns a different person a six-digit code. Names are sampled from seven preselected, token-compatible aliases, and the four codes have distinct first digits. Every code occurs exactly once in both donor and recipient prompts. Background text consists of complete administrative paragraphs composed from a fixed sentence inventory and shuffled by identity; it is synthetic prose, not a natural-document corpus.

The donor question requests the code assigned to one target person. We construct two recipients using the same target/alternate answer pair. **Value swap** exchanges the codes of the target and an alternate person while keeping the question fixed. **Query selection** leaves the entire document unchanged and asks about the alternate person instead. All four codes remain present under both manipulations. Value swap changes two numerical spans, whereas query selection changes a name in the question; their contrast does not match the amount or location of altered information.

The four facts retain their internal order in two position conditions. In the far view, the cluster is inserted after approximately 10% of the background paragraphs; in the near view, it follows all background paragraphs immediately before the question. Paired views have equal token counts and identical token multisets. Distance is measured from the end of a code's token span to the start of `Question:`. Across the four facts, the confirmation gaps are 7,151-7,332 tokens far and 6-109 tokens near. Moving the cluster changes absolute position as well as source-query distance.

Target and alternate slots are balanced across the 40 identities, with ten targets in each position and each of the twelve ordered target/alternate pairs occurring three or four times. The confirmation uses different, pre-specified fact and question wordings from the eight-identity engineering smoke. This holds out surface forms within the same four families, not the families themselves. Whole background paragraphs are added up to the 8,192-token target without truncation.

### Answer-Replacement And Marker Tasks

The preceding studies use five legacy templates: literal lookup, codename lookup, paraphrased lookup, a person-object relation, and a lookup containing an explicitly rejected distractor. A relevant fact is placed among seeded Latin-word filler, with a separate note identifying an unrelated six-digit number. In the distractor template, that unrelated number also appears in the fact. The donor and recipient differ only by replacement of the relevant six-digit code; unlike the multi-fact recipients, these recipients do not contain the donor target code.

Marker confirmation pairs the same example with and without `[NEEDLE_START]` and `[NEEDLE_END]`. Removing the markers changes prompt length; filler is not regenerated or padded to compensate. Its replacement code is sampled independently subject to distinctness and token-alignment constraints, without enforcing a particular common prefix. The temporal and specificity studies instead pair two controlled corruptions for each identity: one changes the first digit, and the other preserves exactly the first digit while changing the second. Later digits may agree by chance. This permits evaluation of decoding interventions without requiring them to undo an already emitted first-digit error.

The five names describe template variations in numerical lookup, not separate natural-language benchmarks. The 160-identity confirmation and subsequent 40-identity studies use fresh identities but retain these template families.

### Source Proximity And Answer Boundary

The specificity study re-renders the legacy facts with uniform block separators. Its far and near chat views reorder the same filler and fact blocks, preserving token counts and multisets; the fact remains before the fixed distractor note. The code-to-question gap is 7,291-7,327 tokens far and 24 tokens near. A short chat view removes the filler but retains the fact, distractor note, and question, so it changes context length and irrelevant content as well as proximity relative to the far task.

The boundary panel compares the far document as raw text, the identical raw text with one trailing ASCII space, and chat-formatted input. Raw prompts contain 8,179-8,181 tokens; adding the space produces 8,180-8,182. The space is supplied input, not a scored answer token. Candidate answers remain the same bare six-digit strings. This manipulation changes the final prompt position and its activation, not merely the evaluation formula.

The specificity design is not fully crossed. Both corruption strata receive sparse patches in all three chat views. Zero and resampled interventions are applied to clean prompts in the different-first stratum across the three chat and two raw views; there are no sparse donor-patch arms in the raw views. Positive calibration is included in every case. In the multi-fact study, clean-prompt ablations occur only in query-selection cases because the clean prompt is identical across its two corruption types.

### Prompt Preparation And Study Separation

Chat inputs use the checkpoint tokenizer's chat template on one user message with `add_generation_prompt=True`; any template-supplied system text and assistant prefix are retained. Raw inputs bypass that template. Earlier nominal 8K lengths were calibrated on a marked raw prompt before marker removal or chat rendering, whereas the multi-fact budget is calibrated on rendered chat input. We therefore report realized lengths rather than treating all 8K conditions as length-identical.

Before model evaluation, paired prompts are checked for the intended edits, equal donor/recipient token lengths, a common final input token, and stable candidate continuation tokenization. For each scored candidate \(y\), tokenizing \(x+y\) must equal the concatenation of the independently tokenized prompt and answer. The evaluated answers comprise six single-digit tokens. The configured maximum prompt length is 32,768 tokens, separate from the nominal 8K construction target; prompts exceeding that maximum, or whose requested continuations exceed the model-position limit, fail validation rather than being truncated. Construction retries concern code distinctness and token compatibility, not model performance.

Study identities use reserved, disjoint seed schedules. The two 4K raw/chat pilots share ten identities and informed the choice of chat for confirmation. Temporal and specificity engineering smokes contain five identities each; the multi-fact smoke contains eight. These are not pooled with confirmation. Separate resampling identities are also excluded from statistical sample sizes. Samples were set by bounded experiment budgets, not a formal power analysis, and all confirmation identities remain in primary analyses regardless of baseline success.

## 3.3 Head Patching And Temporal Scope

We intervene on the concatenated attention-head outputs immediately **before** each selected layer's output projection, implemented as an `o_proj` input hook. This is an activation replacement, not removal of attention weights, a direct key/value-cache substitution, or a separation of query/key and value pathways. Such intervention choices constrain the conclusions available from activation patching ([Zhang and Nanda, 2024](https://arxiv.org/abs/2309.16042); [Heimersheim and Nanda, 2024](https://arxiv.org/abs/2404.15255)).

Let \(t=1\) denote prediction of the first answer token from the final prompt position. For a continuation prefix \(u_{<t}\), let \(z^d_{\ell h t}\) be head \(h\)'s output slice at layer \(\ell\) in an unpatched donor forward pass conditioned on \((x_i^d,u_{<t})\). Let \(z^{r,I}_{\ell h t}\) be the slice computed by the recipient under intervention history \(I\), immediately before replacement. Patching a head set \(G\) over prediction positions \(\mathcal T\) performs

\[
\widetilde z^{r,I}_{\ell h t}=
\begin{cases}
z^d_{\ell h t}, & (\ell,h)\in G\ \text{and}\ t\in\mathcal T,\\
z^{r,I}_{\ell h t}, & \text{otherwise}.
\end{cases}
\tag{1}
\]

Unselected slices retain their current recipient values, not values restored from an unmodified baseline. The temporal study compares

\[
\mathcal T_{\mathrm{prompt}}=\{1\},\qquad
\mathcal T_{\mathrm{decode}}=\{2,3,\ldots\},\qquad
\mathcal T_{\mathrm{combined}}=\{1,2,3,\ldots\}.
\tag{2}
\]

Sets are restricted to positions actually evaluated. Combined greedy intervention therefore includes the prediction of EOS when it occurs. Marker confirmation uses prompt-only intervention; specificity and multi-fact patching compare prompt-only with combined intervention. Prompt-only denotes the computation at the last prompt token, which can be part of the assistant-generation prefix, not all question tokens or document prefill. Its downstream effects can persist after hooks are removed through altered activations and cached state.

All document prefill is unpatched. Prompt tokens except the last are processed in chunks of 1,024, with cache-length checks, explicit positions, and full causal-context masks. Donor and recipient maintain separate caches. In the temporal and subsequent studies, the donor is advanced at every prediction on the same prefix actually consumed by the recipient branch, even when the patch is inactive. In greedy generation, this prefix contains the recipient's generated tokens, not a supplied target answer. In a teacher-forced branch, it contains that branch's candidate tokens. Thus donor activations at later steps are recomputed under a shared prefix, not replayed from an independently generated donor trajectory.

Each condition and candidate branch receives independent copies of the appropriate caches, and the greedy branch is separate from every scoring branch. The marker-confirmation implementation caches only the donor's final-prompt activation because it intervenes only there; all later steps are unpatched. Runtime checks require each selected layer's hook to fire exactly once per active single-position forward pass. These checks complement, rather than replace, behavioral controls.

## 3.4 Matched Controls And Alternative Interventions

**Matched head sets.** Each focal group has five fixed random control draws in the confirmation studies, matched to its head count in every layer. Sampling excludes the focal group's own heads and occurs without replacement within a layer and draw. Draws can overlap one another, repeat a complete set, or include heads from another focal group. We preserve these draws and their original weights; they are not selected for inactivity. The first pilots use one draw per group. Subsequent studies generate the original complete control inventory before retaining their relevant groups, preserving the same control sets rather than changing their seeds when a group is omitted.

**Zero and resampled replacements.** Both act only at \(t=1\) on the clean prompt. Zero ablation replaces the selected slices with zero vectors. Resampling replaces them with slices captured once at the final prompt position of a separate reference instance from the same template/family and view:

\[
\widetilde z^{c,I}_{\ell h 1}=0
\quad\text{or}\quad
\widetilde z^{c,I}_{\ell h 1}=z^{\mathrm{ref}(i)}_{\ell h,\mathrm{last}},
\qquad (\ell,h)\in G.
\tag{3}
\]

Each identity has one preassigned reference identity with distinct codes, rendered in the corresponding view; it is not repeatedly resampled during decoding. References use a seed offset of 50,000 and remain disjoint from study and smoke identities. Their prompt lengths can differ from the recipient's, and their final position is recorded. Resampling therefore differs from the prefix-aligned donor patch: it injects a task-matched query state once, without a decoding donor. It is not a second form of pure component deletion. The specificity and multi-fact ablation panels use \(H\), \(C\), and their respective controls.

**Self-patches.** For each applicable scope, a prompt supplies its own donor activations for the union of heads used in that case. Clean and recipient self-patches must preserve baseline outputs and checked scores. These verify identity behavior but do not establish that a mismatched donor can transfer an answer.

**Positive donor-reproduction calibration.** Specificity and multi-fact studies additionally replace the entire current-position residual immediately before the model's final normalization at every prediction. If \(r_t^d\) denotes that donor residual and \(g_\theta\) the unchanged final normalization and language-model head, then

\[
r_t^{r}\leftarrow r_t^d,
\qquad
\boldsymbol\ell_t^{\mathrm{positive}}=g_\theta(r_t^d)
=\boldsymbol\ell_t^{d}.
\tag{4}
\]

Here \(\boldsymbol\ell\) is the next-token logit vector, and the donor consumes the same branch prefix as the recipient. We require a maximum absolute logit discrepancy of at most \(10^{-5}\) at every evaluated step, candidate-token score agreement within \(10^{-5}\), and exact agreement with the donor baseline's generated token sequence and stopping condition. The positive control must reproduce the donor's actual behavior, including incorrect answers or budget-limited continuations. It deliberately bypasses upstream differences at an output-side site; successful reproduction calibrates the evaluation pipeline, not a sparse retrieval circuit.

## 3.5 Scores And Generated-Answer Measurements

**Teacher-forced scores.** For a candidate code \(y=(y_1,\ldots,y_K)\), \(K=6\), we compute

\[
L_i(I;y)=\frac{1}{K}\sum_{t=1}^{K}
\log p_\theta^I(y_t\mid x_i^{\mathrm{eval}},y_{<t}),
\qquad
\Delta L_i(I)=L_i(I;y_i^*)-L_i(0;y_i^*).
\tag{5}
\]

Here \(x_i^{\mathrm{eval}}=x_i^r\) for donor patching and the positive control, and \(x_i^{\mathrm{eval}}=x_i^d\) for clean-prompt ablation/resampling; \(I=0\) is the corresponding unmodified baseline. The intervention notation includes the donor/reference and temporal scope. Scores use natural logarithms and exclude EOS. No space, punctuation, or answer label is added to the six target tokens, including in raw-prompt evaluations.

Marker confirmation scores the donor target and recipient-correct code. Temporal and specificity studies also score the explicitly unrelated distractor. Multi-fact evaluation scores all four fact codes; the inherited `decoy` field labels one other genuine fact, not an irrelevant answer. With \(S_i(I;y)=K L_i(I;y)\), we report the target-versus-recipient summed log-likelihood margin and, for multi-fact cases, the strongest-alternative margin:

\[
M_i^{\mathrm{pair}}(I)=S_i(I;y_i^*)-S_i(I;y_i^r),
\qquad
M_i^{\mathrm{all}}(I)=S_i(I;y_i^*)-
\max_{y\in\mathcal A_i\setminus\{y_i^*\}}S_i(I;y),
\tag{6}
\]

where \(\mathcal A_i\) is the four-code candidate set. Strict first rank requires \(M_i^{\mathrm{all}}>0\); ties are not counted as wins. This compares specified continuations rather than restricting the model's output to those candidates.

**Free generation.** At every prediction we take the vocabulary-wide logit argmax, stopping at a model-configured EOS token or the study's 24- or 64-token budget, whichever occurs first. No candidate answer is supplied to this branch. Decoding removes special tokens but retains ordinary whitespace and punctuation. For decoded text \(\widehat y_i(I)\), exact target match and a newly induced target are

\[
E_i(I)=\mathbf 1\{\operatorname{strip}(\widehat y_i(I))=y_i^*\},
\qquad
J_i(I)=E_i(I)\,[1-E_i(0)].
\tag{7}
\]

The change \(\Delta E_i=E_i(I)-E_i(0)\) is a net effect and differs from \(J_i\) when pre-existing targets are lost. We report baseline targets, total patched targets, and newly induced targets separately. Exact agreement with the actual donor is a further check on token IDs and stopping, not a substitute for known-target match.

We extract six-digit runs using `(?<!\d)\d{6}(?!\d)`, excluding runs embedded in longer digit sequences. The sole-number diagnostic requires exactly one extracted number equal to the target. Exclusive correct-number content permits multiple occurrences of the target but no other extracted six-digit number. First-number correctness, any target occurrence, recipient/other-fact matches, and verbatim equality without whitespace trimming are separate readouts. A correct number embedded in an explanation is therefore distinguishable from an exact six-digit response. The text-match definition does not itself require EOS; budget-stopped outputs are retained and identified as censored continuations, not asserted to be complete responses.

**Token-level diagnostics.** To separate answer onset from later score effects, we use

\[
K\,\Delta L_i(I)=\Delta\ell_{i1}(I)+
\sum_{t=2}^{K}\Delta\ell_{it}(I),
\tag{8}
\]

where \(\Delta\ell_{it}\) is the paired change in the log probability of the target's token at position \(t\). The original confirmation stores aggregate and first-token scores, permitting first-versus-remaining decomposition; later studies retain every token score. Later teacher-forced terms condition on supplied candidate prefixes, which may differ from the free-generation trajectory. Prefix-match counts, candidate ranks, and log-probability gains are consequently not treated as complete generated answers.

## 3.6 Estimation And Uncertainty

Before multi-fact confirmation, the protocol specified two principal endpoints, exact target generation and mean target-token log probability, and one principal comparison: combined patching of \(H\) against its five-control mean in far query-selection cases. The remaining task, distance, scope, single-head, and replacement contrasts are reported as secondary. Earlier studies retain their own recorded contrasts; the manuscript does not retroactively designate every analysis as a primary test.

For endpoint \(m\), identity \(i\), and repeated condition \(c\), let \(\Delta m_{ic}(G)=m_{ic}(G)-m_{ic}(0)\), with the appropriate baseline from Section 3.5. For fixed matched groups \(R_1,\ldots,R_5\), define

\[
d_{ic}^{m}=\Delta m_{ic}(G)-\frac{1}{5}\sum_{j=1}^{5}\Delta m_{ic}(R_j),
\qquad
\widehat\theta_c^{m}=\frac{1}{N}\sum_{i=1}^{N}d_{ic}^{m}.
\tag{9}
\]

We average controls within identity before estimating uncertainty. Percentage-point effects equal \(100\widehat\theta_c^{E}\); score effects remain in natural-log units per target token. Paired interactions, such as query selection minus value swap, compare these adjusted effects within the same identity:

\[
\widehat\theta_{a-b}^{m}=\frac{1}{N}\sum_{i=1}^{N}
\left(d_{ia}^{m}-d_{ib}^{m}\right).
\tag{10}
\]

The same construction applies to far-minus-near, combined-minus-prompt, and resampling-minus-zero contrasts. These are differences between explicitly defined conditions; for example, a far-minus-near effect is not automatically attributable solely to distance.

We use a paired percentile bootstrap with 4,000 draws and NumPy generator seed 730022. Within each template family, we resample identities with replacement, preserving that family's original size. Each replicate averages the resulting identity-level contrasts over the full study, and the 2.5th and 97.5th percentiles form the reported interval. Marker modes, corruptions, positions, scopes, and control draws are repeated measures, not additional observations. Intervals are 95% pointwise and conditional on the fixed template families, model, and control draws; they are not multiplicity-adjusted and do not quantify uncertainty over alternative models or head-control samples. A degenerate bootstrap interval for zero observed events does not establish zero population event probability.

The frozen analyses retain baseline failures. Restrictions to identities where both baselines succeed, family/slot breakdowns, per-token decompositions, and output-transition ledgers are identified as post-run diagnostics. The exclusive-content metric was introduced after the raw/chat pilots and before the 160-identity confirmation; the first-number diagnostic was added after specificity outcomes but included before multi-fact execution. We do not relabel these additions as endpoints fixed before all earlier runs. Between-study comparisons use disjoint identity batches and changed task constructions and are not treated as paired causal contrasts.

## 3.7 Validation And Reproducibility

Each run retains full prompts, answer candidates, the reserved identity schedule, exact focal/control groups, source hashes, model revision, tokenizer/chat-template fingerprint, package versions, and generated outputs. Temporal and later runners additionally retain per-candidate token scores and step traces recording prefix tokens, cache positions, intervention activation, and hook checks; the positive-control runs record logit discrepancies. The original marker-confirmation rows retain aggregate/first-token scores and generated token IDs rather than the later full trace format.

All confirmation runs used Python 3.10.21, PyTorch 2.5.1+cu121 with CUDA 12.1, Transformers 4.57.1, Accelerate 1.11.0, and NumPy 2.1.2 on SageMaker. The temporal and subsequent confirmation runners require an audited smoke matching their protocol, experiment sources, recorded package versions, checkpoint, and chunk size. Completed outputs are checked for a nonduplicated condition grid, finite scores, intended prompt changes, donor/reference alignment, self/positive controls, and exact reproduction of saved summaries and paired analyses. Engineering validation checks correct execution, not a desired intervention effect or significance threshold.

CPU tests on small randomly initialized Qwen architectures compare cached evaluation with independent full-prefix computations, check branch isolation and greedy donor prefixes, exercise the dense control including bfloat16 execution, and reject malformed inputs and traces. Separate tokenizer-only checks regenerate the saved prompts and decoded outputs without loading pretrained weights. These tests do not replace the pretrained GPU experiments. The protocols, source revisions, full group definitions, and artifact-to-method mapping are retained in the reproducibility materials; experiments and historical outputs are preserved separately from manuscript authoring.
