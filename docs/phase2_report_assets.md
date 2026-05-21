# Phase 2 Report Assets

The report tables and figures are generated from committed `artifacts_phase2` outputs with:

```bash
.venv/bin/python scripts/build_phase2_report_assets.py
```

Outputs are written to:

```text
artifacts_phase2/report_assets/
```

Open the generated HTML preview for a quick visual review:

```text
artifacts_phase2/report_assets/phase2_report_preview.html
```

Use the generated `README.md` in that folder as the figure/table shortlist for the report. The most important report flow is:

1. `fig_01_role_decomposition_16k.svg`
2. `fig_02_l22h7_generalization.svg`
3. `fig_03_single_head_decomposition.svg`
4. `fig_04_l22h7_attention_activation_alignment.svg`
5. `fig_05_necessity_vs_sufficiency.svg`
6. `fig_06_evidence_matrix.svg`

The primary body table is `tables/table_main_results.md`; the remaining tables provide appendix-level detail and confidence intervals. Tables are emitted as `.csv`, `.md`, and `.tex`.
