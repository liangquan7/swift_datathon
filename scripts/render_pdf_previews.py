import os
import sys
import fitz  # PyMuPDF


def render_pdf(pdf_path: str, prefix: str, out_dir: str = "work/preview_pages", dpi: int = 150) -> None:
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    print(f"{pdf_path}: {len(doc)} pages")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = os.path.join(out_dir, f"{prefix}_{idx:02d}.png")
        pix.save(out_path)
    doc.close()


if __name__ == "__main__":
    target = sys.argv[1]
    prefix = sys.argv[2]
    render_pdf(target, prefix)
