#!/usr/bin/env python3
"""Render a report PDF into per-page PNGs for visual QA."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: render_report_pages.py <input.pdf> <output-dir>")

    pdf = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise SystemExit("pdftoppm not found. Install poppler first.")

    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("page-*.png"):
        existing.unlink()

    prefix = out_dir / "page"
    subprocess.run(
        [pdftoppm, "-png", "-r", "144", str(pdf), str(prefix)],
        check=True,
    )

    pages = sorted(out_dir.glob("page-*.png"))
    print(f"Rendered {len(pages)} pages to {out_dir}")


if __name__ == "__main__":
    main()
