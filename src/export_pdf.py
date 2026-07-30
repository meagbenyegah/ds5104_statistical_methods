"""Export notebooks/DS5104_Final_Assessment_Solution.ipynb to PDF.

Renders via headless Chromium (not nbconvert's default LaTeX pipeline, which
this environment doesn't have installed) so long code lines wrap instead of
clipping at the page edge. Code cells are hidden (--no-input) so the PDF is
the write-up plus computed results/charts, not the source.

Usage:
    python3 -m playwright install chromium   # once
    python3 src/export_pdf.py
"""
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "DS5104_Final_Assessment_Solution.ipynb"
PDF_OUT = NOTEBOOK.with_suffix(".pdf")

WRAP_CSS = """
<style>
  .highlight pre, .jp-CodeMirrorEditor pre, div.highlight pre {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-wrap: anywhere !important;
  }
  @media print {
    .jp-Cell { page-break-inside: avoid; }
  }
</style>
"""


def main():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                "python3", "-m", "jupyter", "nbconvert",
                "--to", "html", "--embed-images",
                "--no-input", "--no-prompt",
                str(NOTEBOOK), "--output-dir", tmp,
            ],
            check=True,
        )
        html_path = Path(tmp) / (NOTEBOOK.stem + ".html")
        html = html_path.read_text(encoding="utf-8")
        html_path.write_text(html.replace("</head>", WRAP_CSS + "</head>"), encoding="utf-8")

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{html_path}", wait_until="networkidle")
            page.pdf(
                path=str(PDF_OUT),
                format="A4",
                print_background=True,
                margin={"top": "0.6in", "bottom": "0.6in", "left": "0.5in", "right": "0.5in"},
            )
            browser.close()

    print(f"wrote {PDF_OUT}")


if __name__ == "__main__":
    main()
