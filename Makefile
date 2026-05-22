.PHONY: report-latex report-pdf report-pages report-clean

report-latex:
	python3 scripts/build_latex_report.py

report-pdf: report-latex
	mkdir -p paper/build
	tectonic --keep-logs --keep-intermediates --outdir paper/build paper/main.tex
	cp paper/build/main.pdf paper/retrieval_head_atlas_report.pdf

report-pages: report-pdf
	python3 scripts/render_report_pages.py paper/build/main.pdf paper/rendered_pages

report-clean:
	rm -rf paper/build paper/rendered_pages paper/generated paper/figures
