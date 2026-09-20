# Retrieval Head Atlas: Publication Strategy

Researched on 20 September 2026. This is a strategy proposal, not a record of a submission or acceptance. Ashish confirmed that no ICLR 2027 abstract was registered.

## Recommendation

Prepare a focused research manuscript, post a defensible first version on arXiv, and target Transactions on Machine Learning Research (TMLR) for peer review. Set up the author identity and Google Scholar profile alongside manuscript preparation. Consider a relevant non-archival workshop later for feedback and visibility if its policies are compatible with the journal submission.

This recommendation balances time to a public research artifact, venue fit, cost, and the work's present evidence. A conference submission is an alternative route, not an additional simultaneous archival submission.

The current project provides a promising empirical foundation. Publication readiness still depends on clarifying the contribution against prior work and checking a small set of plausible alternative explanations. Neither the capstone report nor the extended report should simply be uploaded unchanged.

## What We Currently Have

The assessment reviewed the capstone report, relevant sections of the extended report, the results log, consolidated numerical tables, head-group definitions, and the prompt/scoring/patching implementation. It did not rerun GPU experiments or complete an independent audit of every artifact.

- One principal model: Qwen2.5-1.5B-Instruct.
- Five templated retrieval variants, each returning a six-digit answer.
- Two context lengths, approximately 8K and 16K, and three answer positions.
- Forty examples per main length/position setting, with eight examples per variant.
- Ablation, attention tracing, answer substitution, output-slice patching, activation differences, and interaction controls.
- A repeated separation between heads with high ablation sensitivity and heads with stronger clean-answer transfer under the chosen patch.
- L22H7 supplies the largest observed single-head answer-transfer effect among the tested candidates: about +0.335 to +0.501 mean gold-token log-probability units across the six settings.

Two distinctions must survive every rewrite. The reported 88-94% describes L22H7's effect relative to the tested three-head address-group patch; it is not 88-94% task accuracy, total retrieval ability, or recovery of the full clean-corrupt gap. The reported recovery at 8K/early is about 15% of that gap. Also, a head's weak response to one patch does not establish that it never carries answer information.

These values can be traced to `artifacts_phase2/report_assets/tables/table_main_results.md`, the single-head tables, and the corresponding experiment outputs. The claim should remain tied to the tested model, prompts, intervention site, and metric.

## Venue Shortlist

| Route | Verified timing or process | Assessment for this work |
| --- | --- | --- |
| arXiv | Rolling submissions; quality checks normally take 1-4 days, sometimes longer, after submission. Endorsement and preparation are additional. | First public research milestone. A preprint is citable but has not passed peer review. |
| TMLR | Rolling submission; latest annual report gives a 91-day median submission-to-decision interval for conference-length papers. | Recommended primary archival venue after validation. Budget three months or more for review rather than promising the ideal nine-week target. |
| NAACL 2027 through ARR | ARR submission: 12 October 2026; NAACL commitment: 23 December; acceptance notification: 10 February 2027. | Strong topical alternative if conference publication becomes the priority and the manuscript is ready in time. |
| TACL | Monthly submission rounds on the first; its review process targets roughly six weeks to an initial decision, with possible further revisions. | Credible alternative after a stronger natural-language evaluation. An initial decision is not a promise of acceptance or publication. |
| ICLR 2027 workshop | Workshops are still being selected. The suggested paper deadline is 1 February 2027, with acceptance notifications by 26 February. | Possible later community venue; choose a specific workshop only after its call is published. |
| ICLR 2027 main conference | Mandatory abstract deadline: 18 September 2026, 23:59 AoE. Full paper: 25 September. | Unavailable for this work because no abstract was registered. |

Sources: [arXiv availability](https://info.arxiv.org/help/availability.html), [TMLR annual report](https://docs.google.com/document/d/1EsApC9y08bY2Qv2lQ-1NjTgU8QuAT9E30k44QiuSG5c/edit), [TMLR FAQ](https://jmlr.org/tmlr/faq.html), [NAACL call](https://2027.naacl.org/calls/main_conference_papers/), [TACL process](https://transacl.org/index.php/tacl/about), [ICLR workshop call](https://iclr.cc/Conferences/2027/CallForWorkshops), [ICLR author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

### Why TMLR

TMLR explicitly welcomes careful analysis of learning systems. Its criteria emphasize supported claims and useful, clearly communicated findings rather than requiring a state-of-the-art benchmark result. This fits a bounded interpretability study, provided it yields something researchers can learn beyond a classroom reproduction. It charges no submission or publication fee. It is a distinct journal, not a JMLR acceptance. See [acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html) and [editorial policies](https://www.jmlr.org/tmlr/editorial-policies.html).

Use an approximately 8-10-page main manuscript, with supporting detail in appendices. That is our editorial target, not a TMLR page limit. TMLR allows variable length and notes that main texts above 12 pages can take longer to review. Prepare its official template, an anonymous review PDF, and reproducibility material. An identified arXiv version is allowed. See [author instructions](https://jmlr.org/tmlr/author-guide.html).

### Important Conference Constraints

For ICLR, the missed abstract deadline corresponds to 19 September at 17:29 IST. The remaining full-paper deadline does not permit a new submission. ICLR workshop proposal deadlines are deadlines for organizers, not opportunities to submit this research paper.

ARR has a new capacity policy beginning with October 2026: guaranteed review requires a qualified service contributor; unsupported submissions enter a capacity lottery. A qualified non-author can contribute, so this need not alter authorship. This makes the conference route less predictable for a first-time author. See the [11 September policy announcement](https://aclrollingreview.org/sustainable-reviewing-2026).

There are minor current discrepancies between the [ARR schedule](https://aclrollingreview.org/dates) and NAACL's call: meta-reviews are listed as 17 versus 18 December, and reviewer registration as 14 versus 12 October. If taking this route, complete registration by the earlier date and check the live submission instructions. Both sources agree on the October 12 submission and December 23 commitment deadlines.

The nearby workshops checked do not provide an immediate shortcut: [Interpretability for Discovery at NeurIPS 2026](https://interpretability4discovery.github.io/) closed on September 2, and the [ICML 2026 Mechanistic Interpretability Workshop](https://mechinterpworkshop.com/) has already taken place. This is not a claim that every workshop everywhere is closed.

One useful longer-term alternative is the [NAACL 2027 Student Research Workshop](https://www.aclweb.org/portal/content/first-call-papers-cfp-mentorship-program-student-research-workshop-naacl-2027-0): its call allows research completed while a student, lists a November 16 mentorship submission and January 11 direct submission, and offers archival and non-archival options. It is interesting for feedback, not the fastest publication path.

## Contribution And Related Work

The current bibliography is too narrow for a research submission. Several earlier studies overlap with the broad narrative. The comparison below identifies concrete reading priorities; it is not a claim that an exhaustive novelty review is complete.

| Prior work | Relevant overlap | What our manuscript must distinguish |
| --- | --- | --- |
| [Wu et al., Retrieval Head Mechanistically Explains Long-Context Factuality, ICLR 2025](https://arxiv.org/abs/2404.15574) | Sparse retrieval heads with causal relevance are already established. | Finding a retrieval head in Qwen alone is insufficient as the contribution. |
| [Tang et al., RazorAttention, ICLR 2025](https://arxiv.org/html/2407.15891v1) | Proposes retrieval heads gathering information and other heads processing it. | A general retrieval/support split is not a new idea by itself. Our additional evidence would need to be the controlled causal separation and its limits. |
| [Zhao et al., Understanding Synthetic Context Extension via Retrieval Heads, ICML 2025](https://arxiv.org/html/2410.22316v4) | Connects retrieval heads to synthetic/real task transfer and uses ablation and patching across models trained on different data. | Distinguish our within-model answer substitution and role analysis from their training-data comparison. |
| [Lin et al., Retrieval Heads are Dynamic, ACL 2026](https://aclanthology.org/2026.acl-long.715/) | Retrieval-head behavior changes across generation timesteps. | Query-step results should not become claims of a fixed circuit throughout generation. |
| [CompressKV](https://arxiv.org/abs/2508.02401) | Already uses the concept of semantic retrieval heads. | Adding semantic wording to a retrieval task is not enough to claim novelty. |
| [Zhang and Nanda, Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042) and [Heimersheim and Nanda, How to Use and Interpret Activation Patching](https://arxiv.org/abs/2404.15255) | Corruption choices and metrics affect interpretations of patching. | Demonstrate that the observed distinction is informative beyond the specific corruption and score we chose. |

Three possible paper framings:

1. **A focused mechanistic case study.** Characterize answer-transfer and ablation-sensitive components in one small model. Closest to the present evidence, but broad language about a complete circuit would need stronger support.
2. **An empirical study of what head-importance measurements reveal.** Quantify when ablation rankings and answer-transfer rankings disagree, test alternative explanations, and give an actionable evaluation protocol. Recommended framing, with the case study as evidence. The general fact that these measures can differ is already known; our contribution must be the particular empirical findings and validated lessons for retrieval.
3. **A practical inference method.** Use role information for pruning, KV compression, or reliability prediction. Potential future work, but would require systems baselines and measured gains. It is a larger project than the fastest defensible paper from the current results.

Provisional title: **Retrieval Head Atlas: Answer Transfer and Ablation Sensitivity in Long-Context Retrieval**.

Provisional central claim: In a controlled Qwen2.5-1.5B retrieval study, the heads whose removal most reduces answer scores differ from those that transfer answer-specific information under answer-substitution patching. The manuscript should state the conditions under which this separation holds, what it predicts, and where it fails.

Do not claim that we have fully isolated address computation from content transport: the current intervention is at an attention-output slice, not separate query/key and value pathways. Also distinguish functional grouping from an experimentally established causal graph connecting those groups.

## Minimum Validation Plan

The priority is resolving reviewer objections with a bounded set of experiments. Increasing every existing run indiscriminately would leave the largest objections unanswered.

### 1. Independent Confirmation And Behavioral Meaning

Freeze the selected heads and group definitions before collecting confirmation data. Use disjoint seeds and new prompt templates, not simply a larger N beginning at the existing seed base. The current position runner reuses the default seed schedule across settings, so the six settings are repeated conditions rather than six independent collections of identities.

An initial budget could be 32 fresh examples per variant at 8K (160 examples), followed by a selected 16K confirmation slice. This is a planning starting point, not a power calculation or a requirement for statistical significance. Report per-variant effects and uncertainty with paired resampling; preserve pairing and do not count repeated length/position versions as independent examples.

Add clean and intervened generated-answer accuracy, first-answer-token effects, and clean-versus-corrupt answer preference alongside the existing teacher-forced score. A change in answer likelihood should not be presented as an observed change in task accuracy. Check prompt formatting, token alignment, actual lengths, and intervention boundaries while doing this.

### 2. Marker And Template Dependence

The generator visibly wraps the relevant fact in `[NEEDLE_START]` and `[NEEDLE_END]`, surrounds it with sampled filler words, and uses a small set of wording templates. Test marker-free prompts while retaining span positions as metadata outside the model input. Add multiple plausible facts, genuinely different paraphrases, and ordinary prose distractors. Include a modest natural-document slice if the scope retains claims about semantic retrieval beyond templated key-value lookup.

The question is whether the effect survives when the answer location is less conspicuous. If it does not, that boundary must be reported and the contribution reconsidered rather than hidden.

### 3. Corruption And Task Specificity

Changing only the answer value intentionally holds much of the rest of the task fixed. A head with nearly identical clean and corrupt activations will predictably have little effect when those activations are exchanged. That fact alone does not explain its computational role.

Use complementary prompt pairs: change answer identity while holding the queried relation fixed; change which entity is queried while keeping candidate answers available; and test a matched local or context-independent answer task. These distinguish answer transfer, query selection, and general output/formatting support. Add an appropriate non-zero ablation baseline and randomly sampled controls matched in head count and layer distribution. Existing inactive controls are informative, but controls selected for inactivity on discovery data do not replace independent confirmation.

Test the relevant support/content interactions only where they distinguish competing mechanisms. Existing patch-then-ablate experiments should be reused and assessed before designing additional ones. If support ablation harms all tasks similarly, use the narrower description of general model support. If a directional causal pathway is claimed, add interventions that test that pathway directly.

### 4. A Targeted Second-Model Replication

Recommended before a general journal/conference claim, but not a blanket prerequisite for a clearly scoped single-model preprint. Use another small model, preferably from a different family, and rediscover its heads using separate discovery data. Test the qualitative relationship, not whether the literal index L22H7 recurs.

Start at shorter context with a small pilot to establish performance, memory use, and hook correctness, then run the selected confirmation. The existing A10G-scale instance is a plausible starting point; fit and runtime should be measured. Avoid committing to large-model full-attention sweeps without a pilot.

## Manuscript And Release Plan

Build a separate research manuscript while keeping the university reports as historical deliverables. Integrate the earlier atlas work as background and evidence in one coherent story; remove semester-by-semester framing. Confirm that the university submissions were not prior archival peer-reviewed publications.

Suggested main structure: abstract, introduction and precise contributions, related work, tasks and interventions, three or four principal results, limitations, and conclusion. Put the experiment chronology and supplementary sweeps in appendices. Reuse validated scientific plots, with clear denominators and uncertainty; select three or four figures that carry the argument.

Prepare an identified preprint and an anonymous review version from one source. Include model and dependency versions, configurations, prompt construction, discovery/confirmation split, artifact-to-figure mappings, and runnable reproduction commands. Release the relevant code with the paper and use the arXiv record as the main scholarly link once available.

The arXiv submission should contain the LaTeX sources, bibliography/build dependencies, and figures required to compile; arXiv does not accept a PDF alone when it was generated from TeX source. Prefer `cs.CL` as the initial category for this language-model analysis, with a `cs.LG` cross-list if appropriate; moderators determine final classification. See [submission instructions](https://info.arxiv.org/help/submit/index.html).

arXiv provides moderation, not peer review, and papers need substantive research content. Report significant AI assistance accurately and verify every claim and reference. These requirements are directly relevant to how this project was developed. Human authors remain responsible for the work. See [arXiv moderation and tool-use policy](https://info.arxiv.org/help/moderation/index.html).

Use one archival review route at a time. In particular, TMLR and ARR should not review substantially the same paper simultaneously. Before choosing any workshop, confirm in writing whether it is non-archival and whether both venues allow overlap. An archival workshop paper could foreclose TMLR's route for overlapping results. See [TMLR originality policy](https://www.jmlr.org/tmlr/editorial-policies.html) and [ARR multiple-submission rules](https://aclrollingreview.org/cfp).

## Accounts, Endorsement, And Scholar

Begin identity setup while the paper is being revised so it does not become the final bottleneck.

- **Author information:** confirm contributors, order, corresponding author, current affiliation, and access to the IIT Hyderabad address already in the report. An endorsement or occasional feedback does not itself establish coauthorship.
- **OpenReview:** reuse an existing account if there is one; complete education/employment history and conflicts. Public-email registrations can take up to two weeks to activate. See [registration guidance](https://docs.openreview.net/getting-started/creating-an-openreview-profile/signing-up-for-openreview).
- **ORCID:** register or reuse one persistent researcher identifier and connect it to submissions. Registration is [free](https://info.orcid.org/ufaqs/i-am-a-researcher-can-i-use-orcid-for-free/).
- **arXiv:** confirm account and endorsement eligibility early. Under the current rules, an institutional email alone does not guarantee a first-time submitter's endorsement. A suitable existing author, ideally a knowledgeable faculty contact, may endorse after reviewing the work. See [endorsement rules](https://info.arxiv.org/help/endorsement.html).
- **Google Scholar:** create the author profile under a durable personal Google account, with truthful affiliation and interests. Make it public, verify an eligible institutional email for profile search inclusion, and use manual approval for suggested article additions. Add the actual preprint once available; do not claim papers belonging to namesakes. An unindexed article can be entered manually. See [profile guidance](https://scholar.google.com/intl/en/scholar/citations.html).
- **Indexing:** profile setup and paper indexing are separate. A manual entry does not guarantee search indexing or confer peer-review status. Use a stable, accessible scholarly record and allow time for automated indexing. See [Scholar inclusion guidance](https://scholar.google.com/intl/en/scholar/inclusion.html).

## Proposed Schedule

These are our preparation estimates, conditional on validation results, GPU access, author details, and endorsement. They are not venue commitments or automated scheduled work.

| Window | Deliverable |
| --- | --- |
| First 2-3 working days | Detailed novelty comparison, verification of key implementation/measurement assumptions, fixed confirmation protocol, and author/account inventory. |
| Following 1-2 weeks | Focused confirmation runs, revised core narrative, and a complete preprint candidate. Aim for arXiv readiness in roughly 1-2 weeks if the initial checks hold; allow longer if they change the conclusion. |
| Approximately weeks 3-5 | Targeted model/task replication, final journal manuscript, anonymous supplementary package, and TMLR submission. |
| After submission | Address reviews and revise. Budget approximately three months or more to a TMLR decision based on historical turnaround, with acceptance and final publication still contingent. |

If the conference route becomes preferable, make that decision by the end of September so the October 12 ARR deadline and service-contributor requirement remain realistic. Do not start TMLR review and then silently duplicate the submission at ARR.

If the main controls fail, revise the mechanism claim first. A boundary or negative finding may still interest researchers, but its publication value must be assessed on the evidence; it should not be assumed merely because earlier work took substantial effort.

## Ownership And Immediate Next Step

Codex can handle the literature comparison, experiment implementation, result analysis, manuscript editing, figure production, packaging, submission preparation, and response drafting. Ashish supplies or verifies account access, authorship/affiliation facts, and the human author's scientific judgment and required attestations. Account setup and manuscript submission have not been performed during this strategy pass.

The next concrete deliverable should be a reviewer-style publication-readiness assessment with a precise claim-to-evidence comparison and a small executable confirmation plan. That assessment determines what belongs in the first arXiv version and what should be completed before peer review.

Implementation follow-through: the [readiness audit](publication_readiness_audit_2026-09-20.md), [recomputed evidence record](publication_evidence_audit_2026-09-20.json), and [executable confirmation protocol](publication_confirmation_protocol.md) are now available. The first runner is locally tested and tokenization-validated; its pretrained-model GPU pilot remains pending. It deliberately does not yet implement the later new-template, query-selection, task-specificity, or second-model stages.
