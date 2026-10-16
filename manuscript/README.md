# Research Manuscript

This directory is the authoring workspace for the research paper arising from Retrieval Head Atlas. It is separate from the university reports in `paper/`, their prose in `docs/submission_report_draft.md`, and the slides in `presentation/`. Those deliverables remain unchanged.

## Start Here

Read [the manuscript outline](outline.md) for the central argument, four evidence-backed claims, section plan, figure specifications, tables, and limits on interpretation.

Working title:

**From Answer Likelihood to Answer Selection: Auditing Retrieval-Head Interventions in Qwen2.5**

The paper centers on complete answer switching under the frozen three-head intervention in the new multi-fact tasks. Earlier likelihood gains without complete transfer, answer-boundary tests, and temporal controls explain the conditions under which that result holds. It is a single-model empirical study, not a claim to have recovered a complete retrieval circuit.

## Status

- Claims-and-evidence outline completed on 16 October 2026.
- Next writing step: Methods, then Results, using the outline's claim and figure identifiers.
- Full prose, manuscript-specific figure layouts, bibliography, and LaTeX build are not yet created here.
- This manuscript has not been submitted or published. No author profile or indexing status is implied.

## Evidence And Authoring Rules

- Use the immutable runs in `artifacts_publication/` and their frozen protocols in `configs/` and `docs/`. The [latest analysis](../docs/publication_multifact_analysis_2026-10-16.md) connects them to the earlier work.
- Keep historical discovery separate from held-out confirmation. Keep engineering smokes, repeated conditions, and reference donors out of confirmation sample sizes.
- Treat `outline.md` as the manuscript's current claim hierarchy. Older analysis notes record the conclusions available at the time; their intermediate recommendations are not the final paper's narrative.
- Use neutral group names in prose. Internal labels such as `answer_address` and `non_address_core` identify frozen head sets, not proven computational roles.
- Keep raw likelihood changes, matched-control contrasts, generated target answers, actual donor reproduction, and ordinary task accuracy distinct.
- Draft content here without overwriting, importing generated fragments into, or changing the build targets of either capstone report. Refer to existing evidence in place rather than copying large run directories.
- Build scientific figures from recorded data and editable diagrams. Do not generate numerical plots or purported experimental evidence with image synthesis.

The [publication strategy](../docs/publication_strategy_2026-09-20.md) records the proposed arXiv-first, peer-review-later route. Its venue dates and policies are a historical planning snapshot, not a current submission calendar. Recheck submission requirements and confirm authorship, affiliations, acknowledgments, and disclosures before release.
