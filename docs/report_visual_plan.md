# Report And PPT Visual Plan

Status: planning note. The report should use deterministic charts/diagrams for evidence and generated bitmap visuals only where they improve narrative clarity.

## Evidence Figures Already Built

These should be the main scientific figures in the report because they are generated directly from experiment artifacts.

1. `artifacts_phase2/report_assets/figures/fig_01_role_decomposition_16k.svg`
   - Purpose: show necessity vs sufficiency by functional role.
   - Best location: main Results section after introducing role decomposition.

2. `artifacts_phase2/report_assets/figures/fig_02_l22h7_generalization.svg`
   - Purpose: show L22H7 remains a strong answer-content donor across 8k/16k and positions.
   - Best location: main Results section.

3. `artifacts_phase2/report_assets/figures/fig_03_single_head_decomposition.svg`
   - Purpose: show L22H7 dominates L22H10/L21H11/controls in single-head patching.
   - Best location: head-level decomposition subsection.

4. `artifacts_phase2/report_assets/figures/fig_04_l22h7_attention_activation_alignment.svg`
   - Purpose: connect attention, activation difference, and causal patching.
   - Best location: mechanistic evidence subsection.

5. `artifacts_phase2/report_assets/figures/fig_05_necessity_vs_sufficiency.svg`
   - Purpose: explain why support heads can be necessary without being answer donors.
   - Best location: discussion or end of Results.

6. `artifacts_phase2/report_assets/figures/fig_06_evidence_matrix.svg`
   - Purpose: compact evidence matrix across length and position.
   - Best location: summary/results recap or appendix.

## Deterministic Diagrams To Add

These should be created as SVG/HTML/CSS or Mermaid-style diagrams, not image generation, because labels and arrows must be exact.

1. Retrieval-circuit schematic
   - Long context on the left with answer span highlighted.
   - Query token on the right.
   - Blue address/content path through L22H7 and L22H10.
   - Red/orange support-state path through query-tail and non-address core heads.
   - Output: correct answer probability.
   - Key message: answer transport and support state are separable roles.

2. Evidence stack diagram
   - Layers:
     - semantic prompt variants,
     - attention trace,
     - causal ablation,
     - clean-to-corrupt patching,
     - activation-difference analysis,
     - 8k/16k and position generalization.
   - Key message: the claim is supported by converging evidence, not a single metric.

3. Previous-to-current bridge diagram
   - Previous work: sparse retrieval heads for literal long-context copying.
   - Current work: semantic role-decomposed retrieval circuit.
   - Key message: the project matured from head discovery to circuit explanation.

## Generated Bitmap Visuals To Consider

These can use image generation because they are narrative/atmospheric assets, not evidence.

1. PPT opening visual
   - A clean abstract illustration of a long document stream, a highlighted hidden fact, and a few luminous circuit paths converging into an answer.
   - No small text inside the image; text should be added in PowerPoint.
   - Use for title slide or section divider.

2. Report/PPT concept visual: "retrieval as routing"
   - A polished conceptual image showing a query pulling an answer from a distant context through a sparse circuit.
   - Keep it abstract and technical, not cartoonish.
   - Use only as a visual opener, not as evidence.

3. Section divider visual for "From Heads to Circuits"
   - Before/after composition: scattered attention heads on one side, organized role-decomposed pathway on the other.
   - Again, no embedded text; labels should be added in the deck/report.

## Image-Generation Prompt Seeds

Use these only when we are ready to create actual bitmap assets.

### PPT Opening Visual

Use case: scientific-educational
Asset type: presentation title slide background
Primary request: Create a polished abstract technical visual for a mechanistic interpretability project about long-context retrieval in transformer language models.
Scene/backdrop: dark neutral workspace with a long horizontal document stream fading into distance, one small highlighted answer span far back in the sequence, and sparse glowing circuit paths leading from that span to a query point.
Subject: sparse attention-head pathways and answer retrieval circuit.
Composition: wide 16:9, leave clean negative space on the left for title text, visual weight on the right and center.
Style: premium academic AI research presentation, subtle, precise, high contrast, no cartoon elements.
Avoid: readable text, equations, logos, brand names, crowded nodes, people, excessive gradients.

### From Heads To Circuit Divider

Use case: scientific-educational
Asset type: presentation section divider
Primary request: Create an abstract before-and-after visual showing a transition from scattered attention heads to an organized retrieval circuit.
Scene/backdrop: minimal technical background, left side has many faint disconnected nodes, right side has a small structured pathway with two blue content nodes and several orange support nodes.
Subject: mechanistic interpretability circuit discovery.
Composition: wide 16:9, no text, clear left-to-right progression.
Style: clean research-lab visualization, modern but restrained.
Avoid: text, labels, logos, messy network diagrams, biological brain imagery.
