.PHONY: report-latex report-pdf report-pages capstone-latex capstone-pdf capstone-pages report-clean

report-latex:
	python3 scripts/build_latex_report.py

report-pdf: report-latex
	mkdir -p paper/build
	tectonic --keep-logs --keep-intermediates --outdir paper/build paper/main.tex
	cp paper/build/main.pdf paper/retrieval_head_atlas_report.pdf

report-pages: report-pdf
	python3 scripts/render_report_pages.py paper/build/main.pdf paper/rendered_pages

capstone-latex:
	python3 scripts/build_latex_report.py --draft docs/submission_report_capstone.md --generated-dir paper/generated_capstone

capstone-pdf: capstone-latex
	mkdir -p paper/build_capstone
	tectonic --keep-logs --keep-intermediates --outdir paper/build_capstone paper/capstone_main.tex
	cp paper/build_capstone/capstone_main.pdf paper/retrieval_head_atlas_capstone_report.pdf

capstone-pages: capstone-pdf
	python3 scripts/render_report_pages.py paper/build_capstone/capstone_main.pdf paper/rendered_capstone_pages

report-clean:
	rm -rf paper/build paper/build_capstone paper/rendered_pages paper/rendered_capstone_pages paper/generated paper/generated_capstone paper/figures
