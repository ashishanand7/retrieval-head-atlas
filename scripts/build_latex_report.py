#!/usr/bin/env python3
"""Build generated LaTeX sections from the Markdown report draft.

The Markdown draft stays as the authoring surface. This script creates
reviewable LaTeX fragments used by paper/main.tex and copies report figures
into paper/figures for a self-contained local build.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES_SRC = ROOT / "artifacts_phase2" / "report_assets" / "figures"

PANDOC_FROM = (
    "markdown"
    "+pipe_tables"
    "+tex_math_dollars"
    "+tex_math_single_backslash"
    "+raw_tex"
    "+fenced_code_attributes"
    "+backtick_code_blocks"
    "+smart"
)


FIGURES = {
    "fig_01_role_decomposition_16k.png": (
        "fig:role-decomposition",
        "Role decomposition at 16k context. Support heads are strongly necessary under ablation but weak clean-patch answer donors; address heads are the main answer-content donors.",
    ),
    "fig_02_l22h7_generalization.png": (
        "fig:l22h7-generalization",
        "L22H7 remains the dominant single-head answer-content donor across context length and answer position. Error bars show 95\\% normal-approximation intervals.",
    ),
    "fig_03_single_head_decomposition.png": (
        "fig:single-head-decomposition",
        "Single-head decomposition of answer patching. L22H7 carries most of the transplantable answer signal, while L22H10 is a smaller companion and controls remain small.",
    ),
    "fig_04_l22h7_attention_activation_alignment.png": (
        "fig:l22h7-alignment",
        "Converging evidence for L22H7. The same head attends to the needle, changes with answer identity, and restores clean-answer probability when patched.",
    ),
    "fig_05_necessity_vs_sufficiency.png": (
        "fig:necessity-sufficiency",
        "Necessity and sufficiency separate cleanly. Support heads are necessary without being strong answer donors; address heads are the meaningful content donors.",
    ),
    "fig_06_evidence_matrix.png": (
        "fig:evidence-matrix",
        "Retrieval-circuit evidence matrix across context length and needle position. Rows mix metric types, so the figure should be read row-wise as an evidence map rather than as one shared physical scale.",
    ),
}


def run_pandoc(markdown: str, *, shift: int = -1) -> str:
    command = [
        "pandoc",
        f"--from={PANDOC_FROM}",
        "--to=latex",
        "--wrap=none",
        f"--shift-heading-level-by={shift}",
    ]
    result = subprocess.run(
        command,
        input=markdown,
        text=True,
        check=True,
        capture_output=True,
        cwd=ROOT,
    )
    return result.stdout


def figure_block(filename: str) -> str:
    label, caption = FIGURES[filename]
    return f"""```{{=latex}}
\\begin{{figure*}}[t]
\\centering
\\includegraphics[width=0.94\\textwidth]{{figures/{filename}}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure*}}
```
"""


def inject_figures(markdown: str) -> str:
    lines: list[str] = []
    for line in markdown.splitlines():
        marker = re.match(r"^<!--\s*FIGURE:\s*([^>]+?)\s*-->$", line.strip())
        if marker:
            filename = marker.group(1).strip()
            if filename not in FIGURES:
                raise RuntimeError(f"Unknown figure marker: {filename}")
            lines.append(figure_block(filename))
            continue
        if line.startswith("**Main figure.**") and "fig_01_role_decomposition_16k.svg" in line:
            lines.append(figure_block("fig_01_role_decomposition_16k.png"))
            continue
        if line.startswith("**Main figures.**") and "fig_02_l22h7_generalization.svg" in line:
            lines.append(figure_block("fig_02_l22h7_generalization.png"))
            lines.append("")
            lines.append(figure_block("fig_03_single_head_decomposition.png"))
            continue
        if line.startswith("**Main figures.**") and "fig_04_l22h7_attention_activation_alignment.svg" in line:
            lines.append(figure_block("fig_04_l22h7_attention_activation_alignment.png"))
            lines.append("")
            lines.append(figure_block("fig_06_evidence_matrix.png"))
            continue
        if line.startswith("**Discussion figure.**") and "fig_05_necessity_vs_sufficiency.svg" in line:
            lines.append(figure_block("fig_05_necessity_vs_sufficiency.png"))
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def add_citations(markdown: str) -> str:
    replacements = {
        "Transformers [Vaswani et al., 2017]": "Transformers \\cite{vaswani2017attention}",
        "transformer circuits [Elhage et al.]": "transformer circuits \\cite{elhage2021transformer}",
        "induction heads [Olsson et al.]": "induction heads \\cite{olsson2022induction}",
        "causal tracing and activation patching [Meng et al., 2022]": "causal tracing and activation patching \\cite{meng2022rome}",
        "retrieval heads [Wu et al., 2025]": "retrieval heads \\cite{wu2025retrievalheads}",
        "Qwen2.5 [Qwen Team, 2024]": "Qwen2.5 \\cite{qwen25}",
        "Lost in the Middle [Liu et al., 2024]": "Lost in the Middle \\cite{liu2024lost}",
        "Hugging Face Transformers": "Hugging Face Transformers \\cite{wolf2020transformers}",
        "Recent retrieval-head work directly motivates": "Recent retrieval-head work \\cite{wu2025retrievalheads} directly motivates",
        "Another relevant line of work studies causal tracing and activation patching.": "Another relevant line of work studies causal tracing and activation patching \\cite{meng2022rome}.",
    }
    for old, new in replacements.items():
        markdown = markdown.replace(old, new)
    markdown = markdown.replace(
        "All experiments in the current study use Qwen2.5-1.5B-Instruct,",
        "All experiments in the current study use Qwen2.5-1.5B-Instruct \\cite{qwen25},",
    )
    return markdown


def clean_heading_numbers(markdown: str) -> str:
    return re.sub(r"^(#{1,6})\s+\d+(?:\.\d+)*(?:\.)?\s+", r"\1 ", markdown, flags=re.MULTILINE)


def convert_longtables(tex: str) -> str:
    pattern = re.compile(
        r"\{\\def\\LTcaptype\{none\} % do not increment counter\s*"
        r"\\begin\{longtable\}\[\]\{(?P<spec>.*?)\}\s*"
        r"(?P<body>.*?)"
        r"\\end\{longtable\}\s*\}",
        flags=re.DOTALL,
    )

    def replace(match: re.Match[str]) -> str:
        spec = match.group("spec").strip()
        body = match.group("body")
        body = re.sub(r"^\\end(firsthead|head|foot|lastfoot)\s*$", "", body, flags=re.MULTILINE)
        body = re.sub(r"^\\bottomrule\\noalign\{\}\s*$", "", body, flags=re.MULTILINE)
        body = body.strip()
        return (
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\footnotesize\n"
            "\\begin{adjustbox}{max width=\\textwidth}\n"
            f"\\begin{{tabular}}{{{spec}}}\n"
            f"{body}\n"
            "\\bottomrule\\noalign{}\n"
            "\\end{tabular}\n"
            "\\end{adjustbox}\n"
            "\\end{table*}\n"
        )

    return pattern.sub(replace, tex)


def split_draft(markdown: str) -> tuple[str, str, str]:
    markdown = re.sub(r"^# .+\n+", "", markdown, count=1)
    markdown = re.sub(r"^Status: .+\n+", "", markdown, flags=re.MULTILINE)

    abstract_match = re.search(
        r"^## Abstract\s*\n(?P<abstract>.*?)(?=^## 1\. Introduction\s*$)",
        markdown,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not abstract_match:
        raise RuntimeError("Could not find Abstract section in draft")

    abstract = abstract_match.group("abstract").strip()
    body_plus_appendix = markdown[abstract_match.end() :].lstrip()

    appendix_split = re.search(r"^## Appendix(?: Targets)?\s*$", body_plus_appendix, flags=re.MULTILINE)
    if appendix_split:
        body = body_plus_appendix[: appendix_split.start()].strip()
        appendix = body_plus_appendix[appendix_split.end() :].strip()
    else:
        body = body_plus_appendix.strip()
        appendix = ""

    appendix = re.sub(r"^### Appendix ([A-Z]): (.+)$", r"## Appendix \1: \2", appendix, flags=re.MULTILINE)
    return abstract, body, appendix


def copy_figures(figures_dst: Path) -> None:
    figures_dst.mkdir(parents=True, exist_ok=True)
    for filename in FIGURES:
        source = FIGURES_SRC / filename
        target = figures_dst / filename
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=ROOT / "docs" / "submission_report_draft.md")
    parser.add_argument("--generated-dir", type=Path, default=PAPER / "generated")
    parser.add_argument("--figures-dir", type=Path, default=PAPER / "figures")
    args = parser.parse_args()

    draft = args.draft.resolve()
    generated = args.generated_dir.resolve()
    figures_dir = args.figures_dir.resolve()

    generated.mkdir(parents=True, exist_ok=True)
    copy_figures(figures_dir)

    raw = draft.read_text(encoding="utf-8")
    abstract_md, body_md, appendix_md = split_draft(raw)
    body_md = clean_heading_numbers(body_md)

    body_md = inject_figures(add_citations(body_md))
    appendix_md = add_citations(appendix_md)

    abstract_tex = run_pandoc(abstract_md, shift=-1).strip() + "\n"
    body_tex = convert_longtables(run_pandoc(body_md, shift=-1)).strip() + "\n"
    appendix_tex = convert_longtables(run_pandoc(appendix_md, shift=-1)).strip() + "\n" if appendix_md else ""

    (generated / "abstract.tex").write_text(abstract_tex, encoding="utf-8")
    (generated / "body.tex").write_text(body_tex, encoding="utf-8")
    (generated / "appendix.tex").write_text(appendix_tex, encoding="utf-8")


if __name__ == "__main__":
    main()
