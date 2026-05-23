# Local Report Build

This directory contains the local LaTeX wrapper for the submission report.

- `paper/main.tex` is the ACM-style LaTeX shell.
- `paper/references.bib` contains the bibliography.
- `docs/submission_report_draft.md` remains the main prose authoring source.
- `scripts/build_latex_report.py` converts the Markdown draft into generated LaTeX fragments and copies report figures from the committed report assets.

Build the final PDF with:

```bash
make report-pdf
```

Build the shorter capstone/jury version with:

```bash
make capstone-pdf
```

Render page PNGs for visual QA with:

```bash
make report-pages
make capstone-pages
```

The stable PDF artifacts are written to:

```text
paper/retrieval_head_atlas_report.pdf
paper/retrieval_head_atlas_capstone_report.pdf
```
