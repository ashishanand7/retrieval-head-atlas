# Retrieval Head Atlas: A Role-Decomposed Circuit for Semantic Long-Context Retrieval

Status: content draft. This file is intentionally Markdown-first; LaTeX formatting, citation styling, figure placement, and final table formatting can be handled in a later conversion pass.

## Abstract

Long-context language models can often answer questions about information that appeared thousands of tokens earlier, but task accuracy alone does not explain how this retrieval happens internally. In our previous work, we built a retrieval-head atlas for Qwen2.5-1.5B-Instruct and showed that long-context copying behavior is supported by a sparse set of attention heads: only a small subset of heads reliably pointed to the source answer span, and targeted ablations showed that these heads played a causal role in difficult long-distance retrieval.

In this work, we extend that analysis from literal copying to semantic key-value retrieval. Instead of asking only whether a model can copy a hidden string, we test whether the same internal machinery supports retrieval when the query uses aliases, paraphrases, relational descriptions, and distractor-heavy contexts. We combine semantic retrieval probes with causal ablation, clean-to-corrupt activation patching, single-head decomposition, attention tracing, and activation-difference analysis across 8k and 16k token contexts.

The main finding is that semantic retrieval in Qwen2.5-1.5B-Instruct is not implemented by one undifferentiated set of important heads. It is better described as a role-decomposed circuit. A dominant answer-content head, L22H7, directly attends to the answer-bearing context, changes strongly when the answer identity changes, and restores a large fraction of the correct-answer log probability when patched from a clean run into a corrupted run. A smaller companion head, L22H10, contributes additional answer signal. In contrast, a separate set of non-address support heads is strongly necessary under ablation but does not itself transplant answer identity under clean activation patching. This separation between content transport and support-state machinery remains stable across early, middle, and late needle positions and across 8k-16k contexts.

These results sharpen the original Retrieval Head Atlas from a head-discovery project into a mechanistic account of semantic long-context retrieval. The model appears to route answer identity through a small address/content pathway while relying on broader support heads to make that retrieval usable.

## 1. Introduction

Large language models are increasingly used on long inputs: documents, transcripts, logs, codebases, legal records, and multi-turn conversations. In these settings, a model must often answer a question using a specific fact that appeared far earlier in the context. From the outside, this can look simple: the model either retrieves the right answer or it does not. But from an interpretability perspective, that behavioral view is incomplete. A model can achieve high retrieval accuracy while hiding very different internal mechanisms.

The central question of this project is:

> When a transformer retrieves an answer from long context, which internal components actually carry and use that information?

Our previous submission began answering this question by building a Retrieval Head Atlas. We used synthetic long-context probes where a secret value was hidden inside an 8k-token prompt and the model was asked to reproduce it. The model achieved near-perfect accuracy, but the internal attention patterns were not uniform. A sparse cluster of heads, mostly in later layers, placed attention on the answer span at the moment the model generated the answer. Targeted ablations showed that disabling selected heads could selectively damage far-context retrieval, while matched random-head controls did not reproduce the same drop. This established a useful first result: long-context retrieval is behaviorally robust, but internally sparse and causally localized.

However, the previous work left a deeper question open. Literal copying is only one form of retrieval. Real retrieval often requires semantic matching: the query may refer to an entity through an alias, a paraphrase, a relation, or a description rather than an exact repeated string. A head that helps copy a visible token span may or may not be the same head that supports semantic retrieval. Similarly, an attention head can be important in several different ways: it may point to the source answer, carry the answer content, prepare the query state, stabilize downstream computation, or participate in a support role that is necessary but not answer-specific.

This report therefore shifts from asking “which heads matter?” to asking “what roles do the heads play?” The difference is important. A list of important heads is useful, but a circuit explanation should say how those heads divide labor. In this work, we test whether Qwen2.5-1.5B-Instruct uses a stable, role-decomposed circuit for semantic long-context retrieval.

Our main result is that it does. Across semantic variants, context positions, and context lengths, one head, L22H7, consistently behaves like the dominant answer-content carrier. It attends strongly to the answer-bearing span, its activation changes substantially when the answer changes, and patching this head from a clean prompt into a corrupted prompt restores much of the correct-answer probability. L22H10 acts as a smaller companion. Other support heads are also crucial, but in a different way: ablating them strongly damages performance, while patching them does almost nothing to restore the specific answer identity. This means the retrieval mechanism is not simply “all important heads carry the answer.” Instead, answer transport and retrieval support are separable roles.

## 2. What the Previous Submission Established

The previous submission introduced the core methodology and the first version of the Retrieval Head Atlas. Its goal was to move beyond accuracy and inspect the model’s internal retrieval behavior during a controlled long-context task.

The task was a deterministic needle-in-haystack setup. A six-digit secret was inserted into a long prompt, and the model was asked to output that secret. Two settings were emphasized: a far setting where the secret appeared early in the prompt, and a near setting where it appeared close to the query. The model solved both settings with near-perfect accuracy, which created a ceiling effect: ordinary accuracy could no longer reveal which internal components mattered.

To look inside the model, the previous work defined a retrieval event. At the generation step where the model produced the first answer token, each attention head was checked for whether it placed substantial attention mass on the answer span. A head was treated as retrieval-like when it pointed to the answer span on examples where the model answered correctly.

This mapping produced three important observations.

First, retrieval-like attention was sparse. Most heads did not behave like direct pointers to the answer span. The strongest retrieval-like activity was concentrated in a limited set of heads, especially in later layers.

Second, retrieval behavior was distance-sensitive. Some heads were active in near retrieval but not in far retrieval, while the difficult far setting emphasized a different subset of heads. This suggested that long-distance retrieval was not just the same computation as local recency-based access.

Third, causal testing mattered. The previous work did not stop at attention maps. It used targeted ablation to disable selected heads and compared them against layer-matched random-head controls. In the difficult far setting, ablating the selected heads reduced retrieval performance while the matched random controls did not show the same effect. This supported the claim that the mapped heads were not merely correlated with retrieval; they contributed causally.

The previous submission also explored activation patching and external LongBench-style validation. Those results were useful but less conclusive than the ablation result. In particular, patching showed that many internal sites could improve answer probability, which suggested broad representational redundancy. This became a motivation for the present work: instead of treating all important heads as one group, we needed to separate different causal roles more carefully.

The previous report’s headline result can be summarized as follows. On the 8k-token synthetic retrieval probe, the model solved both far and near retrieval almost perfectly, so ordinary accuracy made the task look solved. The internal analysis told a different story. The strongest retrieval-like heads were sparse and concentrated in late-middle to late layers. When the selected top-k heads were ablated, far retrieval dropped from perfect performance to roughly 94%, while a layer-matched random control stayed at perfect performance. Near retrieval remained robust. This gave the previous submission its central message: long-context retrieval can look effortless at the output level while depending on a small, distance-sensitive internal head ensemble.

The previous patching result was also important, but in a different way. Patching selected heads from a clean run into a corrupted run increased the correct-answer log probability, showing that internal activations could repair the output distribution. However, random layer-matched patch sets sometimes repaired even more strongly than the originally selected top-k set. Rather than treating this as a contradiction, the current report treats it as a clue: necessity and sufficiency may live in different parts of the circuit, and a head can be useful for repair without being the exact component whose ablation most damages retrieval. This is one of the reasons the current work focuses on role decomposition.

In short, the previous submission established the foundation:

- Long-context retrieval can be behaviorally perfect while internally sparse.
- A small group of heads shows retrieval-like attention to the answer span.
- Targeted ablations provide causal evidence that selected heads matter for far-context retrieval.
- Patching results suggest that necessity and sufficiency are not the same thing, so a deeper role-based decomposition is needed.

## 3. Current Study: From Retrieval Heads to Retrieval Circuits

The current study extends the previous work in three ways.

First, it moves from literal copying to semantic retrieval. The new prompts still contain a recoverable answer in a long context, but the query does not always match the context literally. We test five variants: literal, alias, paraphrase, relational, and distractor-heavy retrieval. This lets us ask whether the retrieval heads are merely copying exact strings or whether they support broader semantic key-value retrieval.

Second, it decomposes retrieval into functional roles. We use attention tracing to identify heads that directly attend to the answer span, ablation to test which groups are necessary, clean-to-corrupt activation patching to test which groups carry transplantable answer information, and activation-difference analysis to test which heads actually change when the answer identity changes. These tools deliberately answer different questions. A head can be necessary without being sufficient; it can attend to the answer without carrying a cleanly patchable answer representation; it can change activation without being causally important. The report treats these distinctions as central rather than incidental.

Third, it tests whether the mechanism generalizes across position and length. The core suite is run at 8k and 16k token contexts, with the answer placed early, in the middle, and late in the context. This prevents the main claim from depending on one convenient prompt position or one context length.

The current study is organized around four research questions:

1. Do heads discovered from literal long-context retrieval remain important for semantic retrieval?
2. Can the semantic retrieval mechanism be decomposed into address/content heads and support heads?
3. Is there a single dominant head that carries most of the transplantable answer identity?
4. Does the resulting circuit remain stable across 8k-16k contexts and across different answer positions?

The short answer to all four is yes, with useful nuance. The semantic retrieval circuit is broader than the initial literal-copying shortlist, but it has a stable structure. L22H7 is the dominant answer-content head. L22H10 is a smaller companion. Non-address support heads, especially query-tail and broader core heads, are strongly necessary but weak as clean answer donors. This gives the final report a stronger narrative than the previous submission: we are no longer only mapping retrieval heads; we are explaining how different heads divide labor inside a semantic retrieval circuit.

## 4. Contributions

This work makes the following contributions.

1. It extends the Retrieval Head Atlas from literal copying to semantic key-value retrieval. The experiments show that the relevant heads are not limited to exact string matching; they remain important across alias, paraphrase, relational, and distractor-heavy prompts.

2. It identifies a stable role decomposition inside the retrieval circuit. Direct answer-address heads and non-address support heads behave differently under attention tracing, ablation, patching, and activation-difference analysis.

3. It isolates L22H7 as the dominant answer-content head. Across all tested settings, L22H7 has strong answer/needle attention, large clean-corrupt activation differences, and the largest single-head patch effect.

4. It shows that necessity and sufficiency separate cleanly. Non-address support heads are strongly necessary under ablation but do not meaningfully transplant answer identity when patched, while address/content heads are partially sufficient.

5. It demonstrates stability across context length and answer position. The main role-decomposition result holds across 8k and 16k contexts and across early, middle, and late answer placements.

6. It provides a reusable experimental pipeline and report asset set: semantic probes, functional group ablations, component patching, single-head patching, attention tracing, activation-difference analysis, summary tables, and report-ready figures.

## 5. Report Roadmap

The rest of the report should preserve the same broad conference-style structure used in the previous submission: background, experimental setup, methods, results, discussion, limitations, and conclusion. The scientific story is different, but the reader experience should feel familiar to the university/jury format.

Section 6 will summarize background and related work: attention heads, retrieval heads, long-context retrieval, causal ablation, and activation patching.

Section 7 will describe the experimental setup: model, semantic prompt variants, clean/corrupt prompt pairs, context lengths, answer positions, and metrics.

Section 8 will describe the methods: attention tracing, group ablation, clean-to-corrupt patching, single-head decomposition, activation-difference analysis, and matched controls.

Sections 9-12 will present the results: semantic retrieval heads remain causal, the circuit separates into address/content and support roles, L22H7 dominates answer-content transport, and the pattern generalizes to 16k contexts.

Sections 13-15 will discuss interpretation, limitations, future work, and the final conclusion.

## 6. Background and Related Work

### 6.1 Circuit inspiration and related work

This project is inspired by the mechanistic interpretability view that neural networks can sometimes be understood as circuits: interacting internal components that implement a recognizable computation. In this framing, a model is not only a black-box function from prompt to answer. It is also a collection of internal pathways, some of which can be localized, intervened on, and tested.

The transformer-circuits line of work is especially relevant here. That work studies how attention heads, MLPs, residual streams, and token-level features interact to implement behaviors such as induction, copying, and in-context pattern completion. The key methodological lesson is that individual heads are rarely meaningful in isolation. A head becomes interpretable when we understand what it reads, what it writes, and how its output is used by later components.

Induction-head work is a useful analogy. Induction heads are not merely heads with visually interesting attention patterns; they are components that implement a specific algorithmic behavior: detecting a repeated prefix and predicting the next token from an earlier occurrence. The circuit idea is that a capability can be decomposed into roles, and those roles can be tested mechanistically. Our retrieval-head project starts from a similar intuition, but the behavior is long-context answer retrieval rather than short-pattern induction.

Another relevant line of work studies causal tracing and activation patching. These methods compare clean and corrupted runs of a model and patch internal activations to identify where information relevant to the output is stored or transported. This is the source of our clean-to-corrupt patching setup: if a head carries answer identity, then copying that head’s clean activation into a corrupt prompt should increase the clean-answer probability.

Recent retrieval-head work directly motivates the previous submission and this continuation. It argues that sparse attention heads can play a mechanistic role in long-context factuality by retrieving relevant earlier facts. Our previous report adapted that idea into a controlled Retrieval Head Atlas for Qwen2.5-1.5B-Instruct. The present report extends the idea further: it asks whether retrieval heads form a role-decomposed semantic retrieval circuit, rather than merely a shortlist of heads that point to answer spans.

Long-context evaluation work is also part of the motivation. Studies such as “lost in the middle” show that models’ use of long context can depend strongly on where information appears. This matters for interpretability because a mechanism discovered at one answer position may not generalize. That is why the current experiments test early, middle, and late answer positions and extend from 8k to 16k contexts.

This report therefore sits at the intersection of several related lines of work: Transformers [Vaswani et al., 2017], transformer circuits [Elhage et al.], induction heads [Olsson et al.], causal tracing and activation patching [Meng et al., 2022], retrieval heads [Wu et al., 2025], Qwen2.5 [Qwen Team, 2024], and long-context position-sensitivity studies such as Lost in the Middle [Liu et al., 2024].

### 6.2 Attention heads and long-context retrieval

Transformers process a sequence by repeatedly mixing information across token positions. The attention mechanism is the part of the model that decides, at each layer and token position, which earlier tokens should influence the current representation. In a multi-head attention layer, each attention head computes its own attention pattern and value mixture. This means that a single transformer layer does not have one monolithic way of looking backward through context; it has several parallel attention heads that can specialize in different patterns.

This structure makes attention heads a natural starting point for studying long-context retrieval. If a model answers a question using information that appeared thousands of tokens earlier, one possible internal mechanism is that some attention heads directly attend back to the relevant source span when the answer is being produced. Such heads can be interpreted as pointer-like components: they do not merely store information locally, but connect the current query position to a distant part of the context.

However, attention is not the whole computation. An attention head has at least two distinct aspects: where it attends and what information it writes back into the residual stream. A head can look at the right source position but write information that is not decisive for the output. Conversely, a head may be causally important without directly attending to the answer span, because it prepares the query representation, stabilizes the residual stream, or supports downstream heads. This distinction is central to the present report. We do not assume that “attention to the answer” and “causal responsibility for the answer” are the same thing.

### 6.3 Retrieval heads

Recent interpretability work has argued that long-context factual retrieval can depend on a sparse set of retrieval heads: attention heads that attend to relevant earlier facts and whose intervention affects factual output. This idea is attractive because it gives a concrete mechanism for a capability that otherwise looks diffuse. Instead of saying only that “the model uses the context,” we can ask which heads point to the relevant context and whether those heads matter causally.

Our previous submission followed this retrieval-head framing. It showed that, on a controlled needle-in-haystack task, Qwen2.5-1.5B-Instruct solved long-context copying with near-perfect accuracy while only a small subset of heads showed direct retrieval-like attention. This supported the idea that retrieval behavior can be internally sparse even when the model’s external behavior looks robust.

The present work extends that framing in two directions. First, it studies semantic retrieval rather than only literal copying. In realistic retrieval settings, the query often refers to information indirectly: through aliases, paraphrases, descriptions, relations, or nearby distractors. A mechanism that only copies exact strings would be less interesting than one that supports semantic key-value retrieval. Second, it moves from identifying retrieval heads to decomposing a retrieval circuit. The question is not merely whether some heads matter, but what different heads contribute.

### 6.4 Mechanistic interpretability: from observation to intervention

A common risk in interpretability is over-reading observational evidence. Attention maps can be visually compelling, but an attention map alone does not prove that a head is necessary for the model’s answer. A head may attend to the answer span because the answer is already represented elsewhere, or because attention to that span is correlated with success but not required for it.

For this reason, the project uses interventions. Interventions change internal model components and measure the effect on output probabilities. In this report, the main output quantity is the log probability of the correct answer under a controlled prompt. If an intervention lowers the correct-answer log probability, that suggests the intervened component contributed to the behavior. If an intervention restores the correct-answer log probability in a corrupted setting, that suggests the patched component carries useful information for the behavior.

The important point is that different interventions answer different questions. We use ablation, patching, attention tracing, and activation-difference analysis together because no single tool is sufficient on its own.

### 6.5 Ablation as a necessity test

Ablation disables a selected internal component and measures how much the model’s behavior changes. In the context of attention heads, ablation usually means removing or zeroing a head’s contribution at a particular position or stage of the forward pass. If ablating a head or group of heads reduces the correct-answer log probability, this is evidence that the component was necessary for the original computation.

Necessity should be interpreted carefully. A small ablation effect does not always mean a component is irrelevant. Transformer circuits can be redundant: several heads may perform overlapping roles, so removing one head may have little effect while removing a coordinated group has a large effect. Conversely, a large ablation effect does not necessarily mean the ablated component directly carried the answer. It may have supported the computation in another way.

This is why the report uses both single-head and group-level ablations. Single-head sweeps help identify candidate heads and localize effects. Functional group ablations test whether a role-defined group is necessary as a system. Matched inactive controls are included to reduce the risk that a result is caused merely by ablating many heads in late layers.

### 6.6 Activation patching as a sufficiency test

Activation patching asks a different question. Instead of removing a component, we copy an activation from one run of the model into another run. In this project, the most important patching setup is clean-to-corrupt patching. The clean prompt contains one answer; the corrupt prompt is identical except that the answer value is changed. The corrupt prompt makes the model prefer the corrupt answer. We then copy selected clean activations into the corrupt run and measure whether the model becomes more likely to output the clean answer.

If patching a head increases the clean-answer log probability, that head contains information that is at least partially sufficient to restore the answer in that patching setup. This is a stronger kind of evidence than attention alone because it tests whether the internal state can causally move the output distribution.

At the same time, patching has its own limitations. A successful patch does not prove that a component is uniquely responsible for the behavior; many sites may carry overlapping answer information. A failed patch also does not prove that a component is unimportant; the component may be necessary for setting up the computation while not carrying a cleanly transplantable answer representation at the patched site. For this reason, patching and ablation must be read together.

### 6.7 Why necessity and sufficiency can disagree

One of the main conceptual lessons of this work is that “necessary” and “sufficient” are not the same property. A support component can be necessary because the model needs it to maintain a useful query state, route residual information, or stabilize the computation. But that same component may not carry the identity of the answer. If we patch it from a clean run into a corrupt run, it may not restore the clean answer because there is little answer-specific information to transplant.

The reverse can also happen. A component can be a good patch target because it contains answer-specific information, while being less damaging under ablation because other components can partly compensate when it is removed. In a redundant neural system, ablation and patching are complementary views rather than interchangeable tests.

This distinction is the key reason the present report moves beyond a simple “important heads” list. The results show that some heads behave like answer-content heads: they attend to the answer span, change strongly when the answer changes, and restore answer probability when patched. Other heads behave like support heads: they are strongly necessary under ablation but weak as answer donors under patching. The circuit explanation depends on separating these roles.

### 6.8 Semantic retrieval and position generalization

Long-context retrieval is not a single fixed problem. Retrieval can be easy when the answer appears near the query and harder when it appears far away. It can also change when the query refers to the answer literally versus semantically. A robust mechanistic claim should therefore not depend on one prompt shape or one answer position.

This report evaluates semantic retrieval across five prompt variants: literal, alias, paraphrase, relational, and distractor-heavy. It also tests early, middle, and late answer positions at both 8k and 16k token context lengths. This matters because a head that appears important only at one position or one context length may reflect a local artifact rather than a stable retrieval mechanism.

The goal is not to claim that the discovered circuit is universal across all models or all retrieval tasks. The goal is narrower and more defensible: to show that, within Qwen2.5-1.5B-Instruct and this controlled semantic retrieval family, the same role-decomposed mechanism appears repeatedly across meaningful variations in wording, position, and length.

## 7. Experimental Setup

### 7.1 Model

All experiments in the current study use Qwen2.5-1.5B-Instruct, an instruction-tuned transformer language model with 28 layers and 12 attention heads per layer. This gives 336 attention heads in total. The model is small enough to run repeated mechanistic experiments on a single GPU instance, but large enough to show non-trivial long-context retrieval behavior.

The experiments are implemented in PyTorch using Hugging Face Transformers. The model is loaded through the repository configuration as `Qwen/Qwen2.5-1.5B-Instruct`, with automatic precision selection unless otherwise specified. The main GPU execution path is designed for the SageMaker notebook environment used for the project. Local execution on the Mac is used mainly for code editing, artifact analysis, table generation, and report writing.

### 7.2 Semantic retrieval prompt family

The current experiments use a controlled semantic retrieval task. Each prompt contains a long synthetic document, a marked answer-bearing span, irrelevant filler text, and a final question. The answer is always a six-digit value, and the model is instructed to answer with only the six digits.

The prompt structure is intentionally controlled. The answer span is surrounded by explicit markers, `[NEEDLE_START]` and `[NEEDLE_END]`, which allow the analysis scripts to locate the source span for attention tracing. The surrounding context consists of deterministic filler words. This synthetic setup does not aim to imitate a natural document perfectly; its purpose is to create a clean environment where the answer location, answer value, distractor values, and prompt length are known exactly.

The dataset uses five retrieval variants:

1. **Literal.** The document states the access code directly, and the question asks for the access code.
2. **Alias.** The document assigns a key to a codename, and the question asks for the key associated with that codename.
3. **Paraphrase.** The document describes the answer as a six-digit sequence needed to unlock an archive, and the question asks for that sequence.
4. **Relational.** The document links a person to an object and then gives that person’s badge number; the question asks for the badge number of the person who owns the object.
5. **Distractor-heavy.** The document contains a decoy ticket number and a true access code, and the question asks for the true code associated with a codename.

These variants are designed to test more than exact string copying. In the alias, paraphrase, and relational cases, the model must connect the final query to the relevant fact through meaning rather than through a repeated phrase alone. In the distractor-heavy case, the model must avoid a nearby irrelevant six-digit number.

### 7.3 Prompt length and answer position

Prompts are calibrated to target two context lengths:

- 8k tokens, represented by `target_tokens = 8192`
- 16k tokens, represented by `target_tokens = 16384`

For each target length, the amount of filler text is adjusted so that the final tokenized prompt length is very close to the target. In the committed 16k runs, mean prompt length was approximately 16,384 tokens, with observed minimum and maximum lengths within about one token of the target. This matters because the length-generalization claim should be based on actual long prompts, not on a nominal setting that silently produces shorter inputs.

The answer span is placed at three position fractions:

- `needle_frac = 0.1`: early in the document, far from the final query
- `needle_frac = 0.5`: near the middle of the document
- `needle_frac = 0.9`: late in the document, closer to the final query

This gives six main context settings: 8k/0.1, 8k/0.5, 8k/0.9, 16k/0.1, 16k/0.5, and 16k/0.9. Each setting uses 40 examples: five semantic variants times eight examples per variant.

### 7.4 Clean and corrupt prompt pairs

Activation patching requires paired prompts. For each clean prompt, a corrupt prompt is created by replacing the six-digit answer with a different six-digit value while keeping the rest of the prompt structure fixed. The corrupt answer is selected so that the clean and corrupt prompts have matching token lengths under the tokenizer. This prevents trivial alignment problems during activation patching.

The clean prompt asks for the clean answer, and the corrupt prompt asks for the corrupt answer in the same semantic format. The patching experiment then measures whether copying selected activations from the clean run into the corrupt run increases the model’s log probability for the clean answer. This setup creates a controlled causal question: does the patched component carry information about the clean answer identity?

The patching baseline records three quantities:

- the clean prompt’s log probability for the clean answer,
- the corrupt prompt’s log probability for the clean answer,
- the corrupt prompt’s log probability for the corrupt answer.

The second and third values verify that the corrupt prompt really changes the model’s expected answer. In the main 8k early-position run, for example, the clean answer is much less likely under the corrupt prompt, while the corrupt answer is likely under the corrupt prompt. This creates a meaningful recovery gap for patching.

### 7.5 Head groups

The experiments use several head groupings.

The first grouping comes from the previous retrieval-head atlas and early semantic ablation probes. These heads are used to test whether literal-retrieval heads remain relevant under semantic variants.

The second grouping comes from a broader semantic neighborhood sweep. This sweep includes patch-ranked heads, earlier candidate heads, and heads sampled by random controls that turned out to be more damaging than expected. The purpose of the sweep is to identify a broader semantic retrieval core rather than rely only on the first top-k list.

The final grouping is functional. Heads are grouped according to their observed role:

- **Answer-address heads.** Heads that directly attend to the answer or needle span. The key heads in this group are L22H7, L22H10, and L21H11.
- **Non-address core heads.** Heads that are necessary under ablation but do not directly attend to the answer span.
- **Query-tail support heads.** Heads whose attention behavior concentrates near the query tail and whose ablation strongly affects correct-answer probability.
- **First-token/sink heads.** Heads with sink-like attention patterns that are nevertheless functionally important.
- **Inactive controls.** Layer-matched or role-matched heads selected to test whether effects are specific to the active groups.

These groups are not assumed in advance. They are constructed from the sequence of experiments: semantic ablation, neighborhood sweep, attention tracing, and functional controls.

### 7.6 Main metrics

The report uses log probabilities rather than only exact-match accuracy. This is important because the model often answers correctly in the baseline condition, making accuracy insensitive to smaller causal effects. Log probability provides a smoother measure of how strongly the model prefers the correct answer.

The main metrics are:

- **Baseline log probability.** The mean log probability assigned to the correct answer before intervention.
- **Ablation delta.** The change in correct-answer log probability after disabling a selected head or group. Negative values mean the intervention hurt the correct answer.
- **Patch delta.** The change in clean-answer log probability after patching clean activations into a corrupt prompt. Positive values mean the patch restored some clean-answer probability.
- **Recovery fraction.** The patch delta divided by the available clean-vs-corrupt recovery gap. This normalizes patch strength by how much recovery was possible.
- **Gold attention mass.** The attention mass a selected head places directly on the gold answer tokens.
- **Needle attention mass.** The attention mass a selected head places on the full marked answer-bearing span.
- **Activation relative difference.** The clean-vs-corrupt activation difference normalized by the clean activation norm. This measures how strongly a component changes when the answer identity changes.

Together, these metrics let us distinguish four types of evidence: behavioral preference, causal necessity, causal sufficiency, and answer-specific representational change.

## 8. Methods

### 8.1 Overview

The method follows the same broad philosophy as the previous Retrieval Head Atlas pipeline: first identify candidate heads, then test them with causal interventions. The difference is that the current work does not stop at one head ranking. It uses multiple complementary tests to separate different roles inside the retrieval mechanism.

The full pipeline is:

1. Run a semantic ablation probe on the initial retrieval-head candidates.
2. Expand to a broader semantic neighborhood when random controls reveal additional active heads.
3. Trace attention to identify which heads directly attend to the answer span.
4. Build functional groups: answer-address heads, non-address support heads, query-tail heads, sink-like heads, and inactive controls.
5. Run group ablations to test necessity.
6. Run clean-to-corrupt component patching to test sufficiency.
7. Run patch-then-ablate interaction controls to test whether support heads gate use of the address-head patch.
8. Run activation-difference analysis to measure which heads actually change with answer identity.
9. Run single-head patching to identify the main answer-content donor.
10. Repeat the core suite across context positions and context lengths.

This section describes each step in enough detail for the later LaTeX version to include equations, algorithm boxes, and figure callouts.

### 8.2 Scoring answer probability

Most experiments measure the model’s preference for the correct answer using teacher-forced mean answer log probability. Let a prompt be tokenized as \(x\), and let the gold answer be tokenized as \(y = (y_1,\dots,y_m)\). The score is:

\[
\ell(x, y) = \frac{1}{m}\sum_{t=1}^{m}\log p_\theta(y_t \mid x, y_{<t}).
\]

This score is computed at the final query point and then through the answer tokens under teacher forcing. We use mean log probability rather than total log probability so that scores remain comparable when tokenization differs slightly across six-digit answers.

For interventions, the same score is recomputed with selected internal components modified. The key readout is a change in log probability:

\[
\Delta \ell = \ell_{\text{intervened}}(x,y) - \ell_{\text{baseline}}(x,y).
\]

For ablations, a negative \(\Delta \ell\) means the intervention made the correct answer less likely. For patching, a positive \(\Delta \ell\) means the patch restored some probability for the clean answer.

**Visual hook for LaTeX/PPT.** A small equation callout can show the mean log-probability score beside a one-token-at-a-time teacher-forcing diagram.

### 8.3 Query-step interventions

The strongest current experiments intervene at the query step. This means the prompt is first prefetched into the KV cache, and the intervention is applied when the model processes the final prompt token immediately before scoring the answer. This design asks which components matter at the moment the model must use the long context to answer the final question.

The intervention site is the input to the attention output projection (`o_proj`) for selected heads. In a transformer layer, the attention module produces per-head vectors that are concatenated and passed through the output projection. If the hidden size is \(d\), the number of heads is \(H\), and each head has dimension \(d_h = d/H\), then the vector entering the output projection can be viewed as:

\[
z^{(\ell)} = [z^{(\ell)}_1; z^{(\ell)}_2; \dots; z^{(\ell)}_H].
\]

Here \(z^{(\ell)}_h\) is the slice corresponding to head \(h\) in layer \(\ell\). Ablation and patching operate on these slices.

This intervention target is useful because it directly modifies what an attention head writes into the residual stream after it has computed its attention pattern and value mixture. It is also a limitation: it does not separately isolate query, key, value, attention-logit, or value-vector mechanisms. That limitation is discussed later.

### 8.4 Head ablation

For a selected set of heads \(S = \{(\ell,h)\}\), ablation zeros the corresponding head slices at the intervention site:

\[
z^{(\ell)}_h \leftarrow 0 \quad \text{for all } (\ell,h)\in S.
\]

The ablation effect is:

\[
\Delta_{\text{ablate}}(S) = \ell_{\text{ablate}(S)}(x,y) - \ell_{\text{base}}(x,y).
\]

A more negative value means that the selected heads were more necessary for the model’s correct-answer probability. The report uses both single-head ablations and group ablations. Single-head ablations help identify candidate heads. Group ablations test whether a functional subsystem is necessary as a coordinated unit.

The main functional group ablations are run query-only. This keeps the interpretation focused: we are asking what happens when a head’s contribution is removed at the final retrieval/use step, rather than throughout every answer-token continuation step.

### 8.5 Layer-matched and inactive controls

Ablating heads in late layers can have different effects from ablating heads in early layers. Therefore, random controls must be matched by layer. If the active group contains \(k_\ell\) heads in layer \(\ell\), the layer-matched random control also samples \(k_\ell\) heads from the same layer, excluding the active heads when possible. This prevents an unfair comparison where the active group appears important only because it occupies sensitive layers.

The project uses two kinds of controls:

1. **Layer-matched random controls.** These are used in the early semantic ablation probe to test whether the initial top-k heads are more damaging than random heads from the same layers.
2. **Inactive functional controls.** These are used in later group experiments. They are chosen to match the rough layer/role structure of active groups but are empirically weak or inactive in the relevant prior sweeps.

The inactive controls matter because some active groups are large. For example, the non-address core has more heads than the answer-address group. A large ablation effect must therefore be compared against controls that help rule out a simple “more heads removed” explanation.

### 8.6 Semantic ablation probe

The first extension beyond the previous submission tests whether retrieval heads discovered from literal long-context behavior remain important under semantic variants. The probe uses five variants: literal, alias, paraphrase, relational, and distractor-heavy. For each example, the model is scored under three conditions:

- baseline,
- ablation of the selected top-k heads,
- ablation of layer-matched random heads.

For each example, the script records:

\[
\Delta_{\text{topk}} = \ell_{\text{topk ablated}} - \ell_{\text{base}},
\]

and

\[
\Delta_{\text{random}} = \ell_{\text{random ablated}} - \ell_{\text{base}}.
\]

The early run showed that the top-k heads remained consistently damaging across semantic variants, while random controls were weaker on average. However, the random controls were not always inert. Some random draws were unexpectedly damaging. Rather than ignore this, we treated it as evidence that the initial top-k list was incomplete: the random draws were sometimes sampling real members of a broader semantic retrieval circuit.

This led to the neighborhood sweep.

### 8.7 Neighborhood single-head sweep

The neighborhood sweep broadens the candidate set beyond the original top-k heads. It includes:

- heads from the previous patch-ranked and retrieval-event lists,
- heads from the semantic single-head sweep,
- heads from earlier near/far mapping artifacts,
- heads that appeared in damaging random-control draws.

Each head is ablated individually at the query step and ranked by its mean effect on correct-answer log probability. The goal is not to claim that single-head ablation fully explains the circuit. Instead, the sweep is used to discover a broader semantic core and to separate active circuit members from inactive controls.

This step is important narratively. What initially looked like a random-control complication became a discovery mechanism. The semantic retrieval circuit was broader than the first retrieval-head shortlist, and several important support heads were found through this expanded search.

### 8.8 Attention tracing

Attention tracing asks where selected heads look at the query step. The prompt contains explicit marker strings around the answer-bearing span. The tokenizer offset mapping is used to convert character spans into token spans:

- \(G\): the gold answer token span,
- \(N\): the full needle span between `[NEEDLE_START]` and `[NEEDLE_END]`,
- \(D\): distractor six-digit spans outside the gold answer.

For a head with query-step attention vector \(a^{(\ell,h)}\), gold attention mass is:

\[
M_G^{(\ell,h)} = \sum_{i\in G} a_i^{(\ell,h)}.
\]

Needle attention mass is:

\[
M_N^{(\ell,h)} = \sum_{i\in N} a_i^{(\ell,h)}.
\]

The scripts also record distractor mass, gold rank, whether the attention argmax lies inside the gold span, and whether the argmax lies inside the needle span.

Attention tracing is observational, not causal. Its role is to distinguish direct address heads from heads that are necessary for other reasons. In the final interpretation, a direct answer-address head is one that both attends to the answer/needle span and has causal evidence from ablation or patching.

**Visual hook for LaTeX/PPT.** A schematic can show query-step attention split into gold span, full needle span, distractor spans, and remaining context.

### 8.9 Functional group construction

After the neighborhood sweep and attention trace, heads are assigned to functional groups.

The most important groups are:

- **Answer-address.** Heads with direct attention to the answer-bearing span. This group contains L22H7, L22H10, and L21H11.
- **Non-address core.** Heads that are necessary under ablation but do not directly attend to the answer span.
- **Query-tail.** Heads whose attention behavior concentrates near the final query tokens and whose ablation strongly affects retrieval.
- **First-token/sink.** Heads with sink-like behavior that are nevertheless not inert under ablation.
- **Strong/core groups.** Larger aggregate groups used to measure total circuit necessity.
- **Inactive controls.** Heads selected as matched controls for active functional groups.

This grouping is not intended to be a perfect biological taxonomy of heads. It is an operational grouping based on converging experimental evidence. The report uses it to ask whether different roles show different causal signatures.

### 8.10 Component patching

Component patching tests whether selected heads carry answer-specific information that can be transplanted from a clean prompt to a corrupt prompt. Let \(x_c\) be the clean prompt with clean answer \(y_c\), and let \(x_r\) be the corrupt prompt with corrupt answer \(y_r\). The corrupt prompt is constructed so that the clean answer \(y_c\) becomes unlikely.

First, the clean query-step activations are cached:

\[
z^{(\ell,h)}_{\text{clean}}.
\]

Then, during the corrupt run, selected head slices are replaced:

\[
z^{(\ell,h)}_{\text{corrupt}} \leftarrow z^{(\ell,h)}_{\text{clean}}
\quad \text{for all } (\ell,h)\in S.
\]

The patch effect is:

\[
\Delta_{\text{patch}}(S) =
\ell_{\text{patch}(S)}(x_r, y_c) -
\ell_{\text{corrupt}}(x_r, y_c).
\]

The available recovery gap is:

\[
R = \ell_{\text{clean}}(x_c, y_c) -
\ell_{\text{corrupt}}(x_r, y_c).
\]

The recovery fraction is:

\[
\rho(S) = \frac{\Delta_{\text{patch}}(S)}{R}.
\]

If \(\Delta_{\text{patch}}\) is positive, the patched component restored some probability for the clean answer. If \(\rho\) is large, the component restores a substantial fraction of the available clean-corrupt gap.

This is where the role decomposition becomes visible. Address heads have meaningful positive patch effects. Non-address support heads are strongly necessary under ablation but have very small patch effects, suggesting that they do not carry much answer identity at the patched site.

### 8.11 Patch-then-ablate interaction controls

After finding that answer-address patching helps while support-head patching is weak, the next question is whether support heads are required to use the restored address-head signal. To test this, the pipeline runs patch-then-ablate interactions.

For each example, it compares:

- corrupt baseline,
- address-head patch only,
- support-group ablation only,
- address-head patch plus support-group ablation.

The crucial quantity is the patch effect under ablation:

\[
\Delta_{\text{patch under ablation}} =
\ell_{\text{patch+ablate}}(x_r,y_c) -
\ell_{\text{ablate only}}(x_r,y_c).
\]

If this value collapsed relative to patch-only, it would suggest that the support group is required downstream to use the patched address-head information. The refined interaction controls showed a more nuanced result: support-group ablation strongly damages absolute clean-answer log probability, but the address-head patch still provides a positive boost after subtracting the ablate-only baseline. This argues against a strict serial dependency chain and supports a partly additive two-role mechanism.

### 8.12 Activation-difference analysis

Activation-difference analysis measures whether a component changes when the answer identity changes. For a selected group \(S\), the query-step head slices are concatenated into one vector:

\[
v_S = \operatorname{concat}_{(\ell,h)\in S} z^{(\ell,h)}.
\]

For each clean/corrupt prompt pair, we compute:

\[
\Delta v_S = v_{S,\text{clean}} - v_{S,\text{corrupt}}.
\]

The main reported metric is relative L2 difference:

\[
r_S =
\frac{\lVert v_{S,\text{clean}} - v_{S,\text{corrupt}} \rVert_2}
{0.5(\lVert v_{S,\text{clean}} \rVert_2 + \lVert v_{S,\text{corrupt}} \rVert_2)}.
\]

The scripts also record cosine similarity, mean absolute difference, and maximum absolute difference.

This analysis helps explain why necessity and sufficiency differ. Address heads, especially L22H7, have large clean-corrupt activation differences, indicating that they encode answer-specific state. Support heads are often nearly invariant between clean and corrupt prompts, even though ablation shows they are necessary. This supports the interpretation that support heads maintain retrieval/use state rather than carrying answer identity.

### 8.13 Single-head patch decomposition

The answer-address group contains three heads: L22H7, L22H10, and L21H11. Group patching shows that this group carries recoverable answer signal, but it does not reveal whether the signal is distributed evenly. Therefore, the pipeline runs single-head patching for:

- the three answer-address heads,
- several necessary support heads,
- an activation-sensitive but causally weak control,
- inactive controls.

The same patch delta and recovery fraction are computed for each single-head group. This experiment identifies L22H7 as the dominant answer-content donor: it accounts for most of the answer-address group patch effect across the tested settings. L22H10 is a smaller companion. L21H11 attends to the answer and is necessary, but is weak as a standalone clean donor.

### 8.14 Position and length generalization

Once the core role decomposition is established at 8k early-position prompts, the same suite is repeated across:

- early, middle, and late answer positions,
- 8k and 16k context lengths.

The repeated suite includes functional group ablation, functional component patching, single-head patching, attention tracing, and activation-difference analysis. This provides a stronger claim than a single setting would allow. The final report can say not only that L22H7 matters in one probe, but that its attention, activation, and causal patch signatures remain stable across all tested position/length settings.

### 8.15 Statistical reporting

The report uses per-example rows as the basic unit of aggregation. For each metric and setting, the report tables include the mean, standard deviation, and an approximate 95% confidence interval:

\[
\bar{x} \pm 1.96\frac{s}{\sqrt{n}}.
\]

This normal-approximation interval is used as a readable sanity check rather than as a claim of exact distributional normality. For the main results, the direction and consistency of effects are more important than any single p-value. For example, L22H7 patching is positive with confidence intervals above zero in every tested length/position setting, and non-address core ablation is negative in 40/40 examples at every tested setting.

**Visual hook for LaTeX/PPT.** The Methods section should include a compact pipeline diagram: semantic prompts → attention trace → ablation → patching → activation difference → generalization. The Results sections then map directly onto this pipeline.

## 9. Results I: Semantic Retrieval Uses A Causal Head Neighborhood

The first question was whether heads discovered in the previous literal-retrieval setting remain important when retrieval becomes semantic. If the answer were no, then the previous atlas would be mainly a literal-copying result. If the answer were yes, then the same head neighborhood would be involved in a broader form of key-value retrieval.

### 9.1 Initial semantic ablation probe

The initial semantic probe used five variants: literal, alias, paraphrase, relational, and distractor-heavy. Each variant had eight examples, giving 40 examples total. The intervention ablated the selected top-k heads and compared them against layer-matched random heads. The most informative version used query-only intervention, because it focused the ablation on the final retrieval/use step.

The top-k ablation was consistently harmful. Across all 40 examples, the mean query-only top-k ablation effect was approximately:

\[
\Delta_{\text{topk}} \approx -0.389.
\]

A single layer-matched random draw was much weaker, near:

\[
\Delta_{\text{random}} \approx -0.032.
\]

This first result already showed that the selected heads were not merely literal-copy heads. They remained causally relevant when the query used aliases, paraphrases, relational descriptions, and distractor-heavy wording.

The variant-level pattern is important because it shows breadth rather than one lucky aggregate. In the query-only run, the top-k ablation hurt every variant:

| Variant | Top-k ablation delta | Random-k ablation delta |
| --- | ---: | ---: |
| Literal | -0.406 | +0.025 |
| Alias | -0.356 | -0.075 |
| Paraphrase | -0.518 | -0.042 |
| Relational | -0.283 | -0.039 |
| Distractor-heavy | -0.383 | -0.027 |

The paraphrase case was especially strong, with a mean top-k delta around -0.518. This matters because paraphrase retrieval is less like exact copying than the literal condition. The top-k heads were therefore involved in more than just matching repeated surface text.

**Table hook.** In the LaTeX version, this should become the first Results table: “Semantic ablation probe across retrieval variants.” It can report baseline log probability, top-k delta, random-k delta, and top-k minus random gap.

### 9.2 Stronger random controls revealed a broader circuit

The next control used 20 layer-matched random draws instead of a single random draw. This gave a more honest estimate of how often random heads from the same layers could damage retrieval.

The result was nuanced. The top-k group still hurt more than the average random draw:

\[
\Delta_{\text{topk}} \approx -0.389,\qquad
\mathbb{E}[\Delta_{\text{random draw}}] \approx -0.159.
\]

The average gap was therefore:

\[
\Delta_{\text{topk}} - \mathbb{E}[\Delta_{\text{random draw}}] \approx -0.231.
\]

The top-k group hurt all 40 examples, and it was more damaging than each example’s mean random draw in all 40 examples. This supports the original claim that the selected heads are meaningfully important.

However, the random-draw distribution was not fully inert. The most damaging random draw had a mean delta around -0.436, which was even more damaging than the top-k group. At first glance, this could look like a weakness in the result. But inspection showed that the damaging random draws often sampled heads from the same retrieval neighborhood, including heads that later appeared as active circuit members.

This changed the interpretation. The random controls were not simply noise; some of them were accidentally sampling real semantic-retrieval heads outside the initial top-k set. The correct conclusion was not that the top-k heads were unimportant, but that the initial top-k list was incomplete.

### 9.3 Single-head semantic sweep

To resolve this, we ran a query-only single-head ablation sweep over candidate heads. The sweep measured each head’s individual effect on semantic retrieval and ranked heads by mean log-probability damage.

The first patch-ranked single-head sweep identified several individually harmful heads, including:

| Rank | Head | Mean ablation delta | Negative examples |
| ---: | --- | ---: | ---: |
| 1 | L21H11 | -0.110 | 35/40 |
| 2 | L22H7 | -0.107 | 37/40 |
| 3 | L18H3 | -0.087 | 40/40 |
| 4 | L20H9 | -0.055 | 35/40 |
| 5 | L17H3 | -0.043 | 32/40 |

This sweep already showed that semantic retrieval was not driven by a single head. Several heads were consistently harmful when ablated, and their effects varied by semantic variant.

It also separated necessity from earlier patch rankings. Some heads that had appeared useful under patching were not strongly necessary as individual query-step ablations. This was an early warning that the final explanation would need separate notions of necessity and sufficiency.

### 9.4 Neighborhood sweep

The final selection run widened the sweep to 61 heads. This neighborhood included:

- the semantic single-head candidates,
- the patch-ranked candidates,
- heads from the previous near/far retrieval maps,
- non-zero ablation candidates from the earlier work,
- heads sampled by the 20 random draws.

This larger sweep produced 2,440 single-head ablation rows: 61 heads times 40 examples. The most harmful heads were:

| Rank | Head | Mean ablation delta | Negative examples | Read |
| ---: | --- | ---: | ---: | --- |
| 1 | L20H7 | -0.148 | 39/40 | strongest newly discovered support head |
| 2 | L21H11 | -0.110 | 35/40 | answer-address candidate |
| 3 | L22H7 | -0.107 | 37/40 | answer-address/content candidate |
| 4 | L18H3 | -0.087 | 40/40 | broad support head |
| 5 | L17H10 | -0.080 | 40/40 | broad support head |
| 6 | L20H8 | -0.075 | 37/40 | support/query-related |
| 7 | L22H0 | -0.073 | 40/40 | support/query-related |
| 8 | L22H4 | -0.069 | 40/40 | support/query-related |
| 9 | L21H0 | -0.068 | 40/40 | support/sink-like |
| 10 | L20H6 | -0.055 | 40/40 | support head |
| 11 | L20H9 | -0.055 | 35/40 | semantic support |
| 12 | L21H3 | -0.054 | 40/40 | support/sink-like |
| 13 | L21H1 | -0.052 | 40/40 | support/sink-like |

The strongest head in this broader sweep was L20H7, which was not the main answer-address head. This was a major clue. Semantic retrieval depended on a broader late-middle-layer circuit, not only on heads that directly pointed to the answer span.

The sweep also explained the earlier random-control complication. The effects of the 20 random draws correlated strongly with the sum of the single-head effects of the heads they sampled:

\[
r \approx 0.92.
\]

This means the damaging random draws were predictable from the active heads they accidentally included. The random controls were not mysterious; they revealed that the semantic retrieval circuit extended beyond the first top-k group.

### 9.5 Interpretation

The first result section establishes the starting point for the rest of the report:

1. Heads from the previous retrieval atlas remain causally relevant under semantic retrieval variants.
2. The semantic circuit is broader than the initial literal-copying top-k list.
3. Some random-control heads were damaging because they were not truly inactive; they were undiscovered circuit members.
4. The broader semantic core is concentrated mainly in layers 20-22, with support from neighboring layers such as 17, 18, and 23.
5. The strongest necessary heads are not all direct answer-address heads, motivating a functional decomposition.

This last point is the bridge to the next result. If all important heads simply attended to the answer span, the story would be straightforward: retrieval heads point to the answer and carry it forward. Instead, the most harmful single-head ablation in the neighborhood sweep was L20H7, which later attention tracing showed was not a direct answer-address head. Therefore, semantic retrieval requires at least two kinds of machinery: heads that address the answer and heads that support the retrieval computation without directly pointing to the answer.

**Figure hook.** This section could use a small “neighborhood discovery” figure in the report or PPT: initial top-k heads on the left, random-draw contamination in the middle, expanded semantic core on the right. This should be a deterministic diagram rather than a generated image because the head labels matter.

## 10. Results II: The Circuit Splits Into Address Heads And Support Heads

Draft target: introduce the central role decomposition.

Core points:

- Some heads directly attend to the answer span: especially L22H7, L21H11, and L22H10.
- Many necessary heads do not attend to the answer span; they attend to query-tail or sink-like positions.
- Group ablation shows both address and non-address groups matter, but non-address support groups are often more damaging under ablation.
- Matched inactive controls rule out a simple layer/group-size explanation.

Likely figures/tables:

- `fig_01_role_decomposition_16k`
- `table_functional_ablation`

## 11. Results III: L22H7 Is The Dominant Answer-Content Head

Draft target: present the strongest mechanistic result.

Core points:

- Address-head patching restores correct-answer probability; support-head patching is tiny.
- Single-head patching shows L22H7 accounts for most of the answer-address patch effect.
- L22H10 is a smaller companion head.
- L21H11 attends to the answer and is necessary but is weak as a standalone clean donor.
- L20H5 is useful as an activation-sensitive but causally weak control.

Likely figures/tables:

- `fig_02_l22h7_generalization`
- `fig_03_single_head_decomposition`
- `table_functional_patching`
- `table_single_head_patching`

## 12. Results IV: Attention, Activation, And Causality Converge Across 8k-16k Contexts

Draft target: show robustness and convergence of evidence.

Core points:

- L22H7 has strong needle attention at every tested setting.
- L22H7 has large clean/corrupt activation differences at every tested setting.
- L22H7 patch effect remains positive with confidence intervals that stay above zero at every tested setting.
- Non-address support heads remain strongly necessary but weak as content donors at both 8k and 16k.
- The mechanism is robust but not perfectly position-invariant; the 8k middle-position dip is a useful nuance.

Likely figures/tables:

- `fig_04_l22h7_attention_activation_alignment`
- `fig_06_evidence_matrix`
- `table_main_results`
- `table_attention_activation`
- `table_statistical_checks`

## 13. Discussion

Draft target: interpret what the circuit means in simple but precise language.

Planned subsections:

- From head discovery to circuit explanation
- Why support heads can be necessary without carrying answer identity
- Why attention alone is not enough
- Why activation difference alone is not enough
- What this says about semantic long-context retrieval in Qwen2.5-1.5B-Instruct

Likely figure:

- `fig_05_necessity_vs_sufficiency`

## 14. Limitations and Future Work

Draft target: be honest but not apologetic. The study is strong for this model/task family, but not yet universal.

Planned points:

- Single-model limitation
- Synthetic semantic retrieval prompt family
- Patch site limitation: query-step o-proj-input patching
- Need for larger naturalistic evaluation
- Possible next experiments: another model, naturalistic retrieval task, bootstrap reporting, failure-mode sweep

## 15. Conclusion

Draft target: restate the paper-level claim cleanly.

Core takeaway:

Semantic long-context retrieval in Qwen2.5-1.5B-Instruct appears to use a stable, role-decomposed circuit: L22H7 acts as the dominant answer-content head, L22H10 contributes as a smaller companion, and separate support heads are necessary for retrieval performance without directly transplanting answer identity under clean activation patching.

## Appendix Targets

The previous submission ended with a glossary. We should keep that style because it helps jury readability.

Planned appendices:

- Glossary of mechanistic interpretability terms
- Full artifact/file map
- Full tables emitted from `artifacts_phase2/report_assets/tables`
- Additional prompt examples
- Implementation notes and reproducibility commands
