"""File -> list of (page_number, text). Supports pdf, docx, html, md, txt."""
import io
import os


class UnsupportedFile(Exception):
    pass


def parse_file(filename: str, data: bytes) -> list[tuple[int, str]]:

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, (p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    
    if ext == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(data))
        lines = []
        for p in doc.paragraphs:
            if not p.text.strip():
                lines.append("")
                continue
            style = (p.style.name or "").lower() if p.style is not None else ""

            if style.startswith("heading"):
                lines.append(f"## {p.text}")
            else:
                lines.append(p.text)
        return [(1, "\n".join(lines))]
    
    if ext in (".html", ".htm"):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(data, "html.parser")
        for t in soup(["script", "style", "nav", "footer"]):
            t.decompose()
        for h in soup.find_all(["h1", "h2", "h3", "h4"]):
            h.insert_before("\n\n## ")
        return [(1, soup.get_text("\n"))]
    
    if ext in (".md", ".markdown", ".txt", ".text", ""):
        return [(1, data.decode("utf-8", errors="replace"))]
    raise UnsupportedFile(f"Unsupported file type: {ext}")
