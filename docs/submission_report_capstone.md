# Retrieval Head Atlas: A Role-Decomposed Circuit for Semantic Long-Context Retrieval

## Abstract

Long-context language models can answer questions using information that appeared thousands of tokens earlier, but accuracy alone does not show how the model performs this retrieval internally. In the previous submission, we built a retrieval-head atlas for Qwen2.5-1.5B-Instruct and showed that literal long-context copying depends on a sparse set of attention heads. This submission extends that work from literal copying to semantic retrieval, where the query can use aliases, paraphrases, relational descriptions, and distractor-heavy wording.

We test the model with controlled semantic retrieval prompts at 8k and 16k token contexts. We combine attention tracing, causal head ablation, clean-to-corrupt activation patching, single-head patching, and activation-difference analysis. The main result is that semantic retrieval is not implemented by one undifferentiated group of important heads. Instead, it is better explained as a role-decomposed circuit. One answer-content head, L22H7, directly attends to the answer-bearing span, changes when the answer identity changes, and restores the largest single-head portion of correct-answer probability when patched from a clean run into a corrupted run. A broader set of support heads is strongly necessary under ablation but does not itself transplant answer identity.

The contribution of this work is therefore not just another head ranking. It is a more interpretable account of how a small long-context model routes answer information: answer identity travels through a small address/content pathway, while broader support heads make the retrieval computation usable.

## 1. Introduction

Large language models are now often used on long inputs: reports, transcripts, logs, codebases, legal documents, and long conversations. In these settings, the model may need to answer a question using one small fact that appeared far earlier in the context. From the outside, the model either answers correctly or it does not. But for interpretability, this is not enough. Two models can achieve the same accuracy while using very different internal mechanisms.

The central question of this project is:

> When a transformer retrieves an answer from long context, which internal components actually carry and use that information?

In the previous submission, we studied a literal needle-in-haystack task. A six-digit secret was inserted into a long prompt, and the model had to copy it. The model solved the task with near-perfect accuracy, but the internal picture was sparse: only a small subset of attention heads consistently pointed to the answer span at the moment of answer generation. Targeted ablations showed that these heads had a causal role, especially for difficult far-context retrieval.

That result was useful, but literal copying is only the first step. Real retrieval is often semantic. The query may refer to a fact using an alias, a paraphrase, a relation, or nearby distractors rather than an exact repeated phrase. The current submission asks whether the retrieval-head story still holds in that more realistic setting, and whether the heads have distinguishable roles rather than all doing the same job.

Our answer is yes, with an important refinement. Semantic retrieval in Qwen2.5-1.5B-Instruct is not just a list of heads that matter. It is a small circuit with different roles. Some heads directly address the answer span and carry answer identity. Other heads support retrieval but do not themselves carry the answer in a cleanly patchable form.

## 2. What Changed Since the Previous Submission

The previous submission established three ideas:

1. Long-context retrieval can look behaviorally easy while being internally sparse.
2. Some attention heads behave like retrieval heads by pointing to the answer span.
3. Targeted ablation of selected heads can damage far-context retrieval more than matched random controls.

The current work extends this in three ways.

First, the task is semantic rather than only literal. The query can ask for the answer through literal wording, an alias, a paraphrase, a relational clue, or a distractor-heavy context.

Second, the analysis separates head roles. We do not assume that every important head carries answer content. A head may be necessary because it supports the query state, stabilizes the computation, or helps downstream heads use retrieved information.

Third, we test whether the mechanism remains stable across answer position and context length. The core experiments are run at 8k and 16k token contexts, with the answer placed early, middle, and late in the prompt.

The resulting narrative is a continuation of the previous atlas: we move from "which heads matter?" to "what roles do those heads play?"

## 3. Background

A transformer \cite{vaswani2017attention} uses attention heads to mix information across token positions. Each head computes where to attend and what vector to write back into the model's residual stream. This makes attention heads a natural place to look for long-context retrieval mechanisms: if a model answers using information far earlier in the prompt, some heads may point back to the relevant span.

However, attention patterns alone are not enough. A head can look at the answer without being decisive for the output. A head can also be necessary without directly attending to the answer, because it may prepare or support the retrieval computation. This is why the current work combines several kinds of evidence:

- **Attention tracing:** where does the head look?
- **Ablation:** is the head or group necessary?
- **Activation patching:** can the head carry answer information from a clean run into a corrupted run?
- **Activation difference:** does the head change when the answer identity changes?

This follows the broader mechanistic interpretability idea that neural networks can sometimes be understood as circuits: interacting components that implement a recognizable computation \cite{elhage2021transformer,olsson2022induction}. It also builds on causal tracing and activation patching methods \cite{meng2022rome}, retrieval-head work \cite{wu2025retrievalheads}, and long-context position-sensitivity studies such as Lost in the Middle \cite{liu2024lost}.

## 4. Experimental Setup

All experiments use Qwen2.5-1.5B-Instruct \cite{qwen25}, a small instruction-tuned transformer model. The smaller model size makes the experiments feasible on a single AWS ml.g5.xlarge GPU instance while still allowing full attention-head analysis.

Each prompt contains a marked answer-bearing "needle" span inside a long context. The model is asked a final question whose answer is a six-digit value in the needle. The prompt family has five semantic variants:

- **Literal:** the query directly matches the context wording.
- **Alias:** the query uses an alternate name for the target.
- **Paraphrase:** the query asks the same fact with different wording.
- **Relational:** the query describes the target through a relation.
- **Distractor-heavy:** the prompt includes many nearby distractor values.

Most scores use teacher-forced mean answer log probability. If the prompt is $x$ and the gold answer tokens are $y_1,\dots,y_m$, the score is:

\[
\ell(x,y) = \frac{1}{m}\sum_{t=1}^{m}\log p_\theta(y_t \mid x, y_{<t}).
\]

For ablation, a negative change in this score means the intervention made the correct answer less likely. For patching, a positive change means the patch restored probability for the clean answer.

The main interventions are applied at the query step, just before the model begins scoring the answer. This focuses the analysis on the moment when the model must use the long context to answer the question.

## 5. Methodology

The experiment pipeline has five stages.

**Stage 1: semantic ablation probe.** We first test whether heads from the previous retrieval atlas still matter under semantic variants. We ablate the selected top-k heads and compare them with layer-matched random heads.

**Stage 2: broader head sweep.** Some random controls were more damaging than expected. Instead of ignoring this, we used it as a clue: the random draws were sometimes sampling real circuit members outside the original shortlist. We therefore ran a broader single-head sweep over a larger candidate neighborhood.

**Stage 3: attention tracing.** We measure how much query-step attention each selected head places on the gold answer span, the full needle span, and distractor spans. This identifies direct answer-address heads.

**Stage 4: ablation and patching by functional group.** Heads are grouped into answer-address heads, non-address support heads, query-tail support heads, sink-like heads, and inactive controls. Ablation tests necessity; clean-to-corrupt patching tests whether the group carries transplantable answer information.

**Stage 5: single-head and generalization analysis.** We decompose the answer-address group head by head and repeat the core tests across 8k/16k contexts and early/middle/late answer positions.

This pipeline deliberately separates different questions. Attention tells us where a head looks. Ablation tells us whether the computation needs it. Patching tells us whether it carries answer-specific information. A convincing mechanism should be supported by all three, but not every important head needs to score highly on every test.

## 6. Results

### 6.1 Semantic retrieval uses a causal head neighborhood

The first result is that the previous retrieval heads remain causally relevant when the task becomes semantic. In a query-only ablation run with 40 examples, top-k ablation hurt every semantic variant.

| Variant | Top-k ablation delta | Random-k ablation delta |
| --- | ---: | ---: |
| Literal | -0.406 | +0.025 |
| Alias | -0.356 | -0.075 |
| Paraphrase | -0.518 | -0.042 |
| Relational | -0.283 | -0.039 |
| Distractor-heavy | -0.383 | -0.027 |

The paraphrase result is especially important because it is less like direct copying. This shows that the selected heads are involved in semantic retrieval, not only exact string matching.

The stronger random-control run added nuance. The top-k group was more damaging than the average random draw, but some random draws were also damaging. A broader single-head sweep explained why: those random draws had accidentally sampled additional active support heads from the same late-layer neighborhood. This moved the project from a narrow shortlist to a broader semantic retrieval circuit.

### 6.2 The circuit separates into address heads and support heads

Attention tracing identified three direct answer-address heads: L22H7, L22H10, and L21H11. These heads place substantial attention on the answer-bearing span at the query step. L22H7 is the cleanest example: in the 8k early-position setting, it places 0.665 attention mass on the gold answer tokens and 0.848 on the full needle span.

But the most damaging support heads are not always answer-address heads. L20H7, for example, is highly necessary under ablation but places almost no direct attention on the answer span. This tells us that semantic retrieval needs more than direct answer pointing.

The functional group ablations make the split clear:

| Group | Heads | Mean ablation delta | Negative examples |
| --- | ---: | ---: | ---: |
| Answer-address | 3 | -0.163 | 39/40 |
| Non-address core | 16 | -1.269 | 40/40 |
| Query-tail support | 9 | -1.015 | 40/40 |
| First-token/sink | 6 | -0.376 | 40/40 |
| Address inactive control | 3 | +0.011 | 13/40 |
| Query-tail inactive control | 9 | +0.024 | 19/40 |

Support heads are therefore not decorative. Removing them strongly damages correct-answer probability. But this still does not mean they carry the answer itself.

<!-- FIGURE: fig_01_role_decomposition_16k.png -->

### 6.3 Necessity and answer-content sufficiency are different

The strongest conceptual result is the separation between necessity and sufficiency. Non-address support heads are strongly necessary under ablation, but their clean activations do not transplant the clean answer into a corrupted prompt.

At 8k early position:

- Answer-address patching: +0.508 log-probability units
- Non-address core patching: +0.035 log-probability units
- Query-tail support patching: +0.013 log-probability units

At 16k, the same pattern holds. Answer-address patching remains large, while support-head patching remains tiny. This means support heads are important, but not because they directly carry the answer identity in this patching setup.

This distinction is important for interpretability. If we only looked at ablation, we might conclude that the non-address core is "the answer circuit." If we only looked at patching, we might miss the support machinery. The full picture requires both.

### 6.4 L22H7 is the dominant answer-content head

The answer-address group contains L22H7, L22H10, and L21H11. Single-head patching shows that the answer signal is not evenly distributed across them.

At 8k early position:

| Head | Patch delta | Recovery fraction | Positive examples |
| --- | ---: | ---: | ---: |
| L22H7 | +0.457 | 15.2% | 38/40 |
| L22H10 | +0.072 | 2.4% | 36/40 |
| L21H11 | +0.016 | 0.6% | 25/40 |

L22H7 alone accounts for roughly 90% of the full answer-address group patch effect in this setting. Across the tested settings, it accounts for roughly 88% to 94% of the group effect.

L22H10 is a smaller companion. L21H11 is address-like and necessary, but weak as a standalone answer donor. This is a useful warning: direct answer attention does not automatically imply that the head carries the main answer identity.

<!-- FIGURE: fig_02_l22h7_generalization.png -->

### 6.5 Evidence converges across 8k and 16k contexts

L22H7 remains positive across every tested length and position:

| Setting | L22H7 patch delta | 95% CI |
| --- | ---: | --- |
| 8k / early | +0.457 | [+0.403, +0.512] |
| 8k / middle | +0.335 | [+0.284, +0.387] |
| 8k / late | +0.462 | [+0.414, +0.509] |
| 16k / early | +0.408 | [+0.337, +0.479] |
| 16k / middle | +0.500 | [+0.431, +0.568] |
| 16k / late | +0.501 | [+0.438, +0.564] |

It also keeps strong needle attention and large clean-corrupt activation differences across these settings. The only notable wrinkle is the 8k middle-position dip, where patching is smaller but still clearly positive. We keep this in the report because real mechanisms are rarely perfectly uniform.

The full technical report includes the complete attention/activation alignment figure and an evidence matrix across all settings. In this shorter capstone version, the main body keeps only the two figures needed for the core story: role decomposition and L22H7 generalization.

## 7. Discussion

The main lesson is that "important head" is not one role. In this project, the heads separate into at least two roles:

- **Answer-address/content heads:** heads that point to the answer-bearing span and carry answer-specific information. L22H7 is the dominant example.
- **Support heads:** heads that are necessary for retrieval but do not themselves transplant the answer identity when patched.

This explains why different interpretability tools can appear to disagree. Ablation finds components the model needs. Patching finds components that carry enough information to repair the answer. These can overlap, but they do not have to be identical.

For the jury, the most important takeaway is simple: we did not only find that the model has retrieval heads. We found that semantic retrieval is organized into roles. One small pathway carries the answer, while other heads support the computation that lets the answer affect the final output.

## 8. Limitations and Future Work

This study is intentionally controlled, which is both a strength and a limitation. Controlled prompts let us know the exact answer span, distractor spans, answer identity, and corrupt-answer identity. That makes mechanistic intervention possible. But the prompts are still synthetic.

The main limitations are:

- The result is specific to Qwen2.5-1.5B-Instruct. Other models may use different head indices or different circuits.
- The patching site is the query-step attention-output slice. It does not separately isolate query, key, value, attention-logit, or MLP mechanisms.
- The main grid uses 40 examples per setting. The effects are consistent, and a paper version would benefit from larger confirmation runs.
- The task family tests semantic key-value retrieval, but not full natural-document reasoning or multi-hop retrieval.

Future work should test whether the same role decomposition appears in other models, whether naturalistic documents show similar answer-content and support roles, and whether more precise patching sites can explain what L22H7 reads and writes internally.

## 9. Conclusion

This submission extends the Retrieval Head Atlas from literal long-context copying to semantic long-context retrieval. The previous work showed that retrieval-like attention is sparse and causally relevant. The current work shows that the mechanism is also role-decomposed.

In Qwen2.5-1.5B-Instruct, semantic retrieval uses a small answer-content pathway dominated by L22H7, with L22H10 as a smaller companion. L22H7 attends to the answer-bearing span, changes when the answer identity changes, and restores clean-answer probability when patched into a corrupted run. A broader set of support heads is strongly necessary under ablation but does not itself carry the clean answer identity under patching.

The final message is:

> Semantic long-context retrieval is not just a diffuse model capability and not just a list of important heads. In this model and task family, it is supported by a role-decomposed circuit: a small address/content pathway carries answer identity, while broader support heads make retrieval work.

## Appendix

### Appendix A: Short glossary

- **Attention head:** one attention component inside a transformer layer.
- **Retrieval head:** a head that attends to answer-relevant context and affects retrieval behavior.
- **Answer-address head:** a head that directly attends to the answer-bearing span.
- **Support head:** a head that is necessary under ablation but does not directly carry answer identity under patching.
- **Ablation:** removing a component to test whether it is necessary.
- **Activation patching:** copying activations from a clean run into a corrupted run to test whether they carry useful information.
- **Teacher-forced log probability:** the log probability assigned to the gold answer tokens when they are supplied to the model one by one.
- **Recovery fraction:** the fraction of the clean-corrupt answer-probability gap restored by a patch.
