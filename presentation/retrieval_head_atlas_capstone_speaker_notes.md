# Retrieval Head Atlas: Capstone Presentation Speaker Notes

Approximate target: 9-10 minute recorded walkthrough.

## Slide 1: Retrieval Head Atlas: semantic retrieval circuit

Good afternoon. This is Ashish from the MDS 2023 batch, and this is the continuation of my capstone project, Retrieval Head Atlas. In the previous submission, I studied literal long-context copying: can a model find a hidden code in a long prompt, and which attention heads help it do that? In this submission, I extend the project to a more realistic question: when the query is semantic, and not just exact copying, what internal circuit carries the answer?

## Slide 2: Accuracy hides mechanism

The motivation is the same as last semester, but the task is harder. A language model can answer from a long context, but accuracy only tells us that the answer came out correctly. It does not tell us whether the model found the exact span, whether it used a broad distributed representation, or whether a small circuit routed the answer. This matters even more for semantic retrieval, where the question might use an alias, a paraphrase, or a relation instead of repeating the exact phrase.

## Slide 3: Previous submission recap

Before I move into the new work, I want to briefly anchor what was already submitted last semester. The first version of Retrieval Head Atlas mapped a literal needle-in-a-haystack task. The main result was that retrieval activity was sparse and concentrated in later layers, mainly around layers 19 to 23. Then ablation showed those suspect heads were not just correlated with retrieval, because removing them selectively hurt the hard long-range setting. Finally, patching showed an important nuance: the system was specialized for necessity, but answer information was also more redundant than the first head list suggested. This is the foundation I build on here.

## Slide 4: From head list to head roles

This submission starts from that foundation rather than replacing it. Last semester the project asked: which attention heads are involved in literal retrieval? This semester I ask the next question: when retrieval becomes semantic, what role does each important head play? Some heads may directly address the answer span, some may support the computation, and only some may carry answer identity strongly enough to repair a corrupted run.

## Slide 5: Experimental setup

The model is still Qwen2.5-1.5B-Instruct, so the comparison remains clean. I use controlled long-context prompts at 8k and 16k tokens, with a marked answer-bearing span placed early, middle, or late in the context. The query comes in five semantic variants: literal, alias, paraphrase, relational, and distractor-heavy. The main score is teacher-forced log probability of the correct answer tokens, which lets us see effects even when final accuracy is high.

## Slide 6: Methodology

The method intentionally reuses the earlier interpretability pipeline. Attention tracing asks where a head looks at the query step. Ablation asks whether the model needs that head or group of heads. Activation patching asks whether clean activations from that head can repair a corrupted prompt and restore probability on the clean answer. The difference is that I now compare these tools to separate head roles rather than collapse everything into one ranking.

## Slide 7: Semantic variants remain causal

The first new result is that the previous retrieval-head neighborhood remains causally relevant when the task becomes semantic. When I ablate the selected retrieval heads at the query step, correct-answer log probability drops for every variant. The paraphrase condition is especially important because it is least like literal copying. This tells us the heads are involved in semantic retrieval, not only exact string matching.

## Slide 8: Random controls became a clue

One interesting moment in the project was that some random controls were more damaging than expected. Instead of treating that as noise, I used it as a clue. A broader sweep showed that those random sets were sometimes sampling additional support heads from the same late-layer neighborhood. So the story expanded: we were not looking at one tiny shortlist, but at a broader semantic retrieval circuit with different kinds of members.

## Slide 9: The circuit splits into roles

After grouping the heads functionally, the role split becomes clear. Answer-address heads directly attend to the answer-bearing span. Support heads do not necessarily point to the answer, but ablating them strongly damages the model. In this result, non-address core heads and query-tail support heads are much more damaging under ablation than the direct address group. That means support heads are not decorative; the model needs them.

## Slide 10: Necessity is not sufficiency

The key conceptual result is that necessity and answer-content sufficiency separate cleanly. Support heads are very necessary under ablation, but when I patch their clean activations into a corrupted run, they barely restore the clean answer. Answer-address heads show the opposite pattern: they are less damaging as a group under ablation, but they carry much more patchable answer identity. This resolves the earlier redundancy result into a clearer role decomposition.

## Slide 11: L22H7 is the dominant answer donor

The answer-address group itself is also not uniform. Single-head patching shows that L22H7 carries most of the transplantable answer signal. L22H10 is a smaller companion, and L21H11 looks address-like but is weak as a standalone answer donor. This is useful because it prevents an overly simple conclusion: direct attention to the answer does not automatically mean the head carries the main answer identity.

## Slide 12: The result generalizes across length and position

The L22H7 result holds across the tested settings. It remains positive at 8k and 16k context lengths, and when the answer is early, middle, or late. There is a small dip at the 8k middle position, but the effect remains clearly positive. The important point is that the same head keeps aligning attention, activation difference, and causal patching evidence across the main grid.

## Slide 13: Synthesis

So the final story is not simply that the model has retrieval heads. The stronger claim is that semantic long-context retrieval is role-decomposed. One small address and content pathway, dominated by L22H7, carries answer identity. A broader support neighborhood is strongly necessary for retrieval to work, but does not itself transplant the answer under this patching setup. This explains why attention, ablation, and patching can appear to disagree unless we separate head roles.

## Slide 14: Limitations and future work

There are still important limitations. The prompts are controlled and synthetic, which is useful for mechanistic intervention but not the same as natural documents. The model is only Qwen2.5-1.5B-Instruct, so other models may use different circuits. The patching site is still head-level attention output; future work should isolate query, key, value, attention logits, and MLP contributions. A paper version should also run larger confirmation grids and test more models.

## Slide 15: References

The work builds on the transformer architecture, mechanistic interpretability as circuit analysis, causal tracing and activation patching, long-context retrieval analysis, and retrieval-head work. These references are the main intellectual background for why we can treat a model's internals as a testable mechanism rather than only a black-box predictor.

## Slide 16: Closing

To close, the contribution of this submission is a clearer mechanism for semantic retrieval. We started with a literal retrieval-head atlas, then moved to semantic prompts and discovered that important heads have different roles. L22H7 is the clearest answer-content head, while other support heads make retrieval possible. So the new submission is not a separate project dropped on top of the earlier one; it is the next layer of the same investigation. Thank you.
