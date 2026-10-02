"""Structure-aware chunker: respects Markdown headings and paragraph boundaries,
splits oversized paragraphs on sentence punctuation (incl. Devanagari danda / CJK stops)."""
import re
from dataclasses import dataclass

sent = re.compile(r"(?<=[.!?।。！？])\s+")


@dataclass
class Chunk:
    text: str
    section: str


def _split_long(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    pieces, buf = [], ""
    
    for sent in sent.split(text):
        while len(sent) > size:  # no punctuation: hard split at whitespace
            cut = sent.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if buf and len(buf) + len(sent) + 1 > size:
            pieces.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        pieces.append(buf)
    return [p for p in pieces if p]


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""
    buf = ""
    buf_section = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(Chunk(buf.strip(), buf_section))
        buf = ""


    for para in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        para = para.strip()

        if not para:
            continue
        if re.match(r"^#{1,6}\s", para):
            first, _, rest = para.partition("\n")
            new_section = first.lstrip("#").strip()
            if new_section != section:
                flush()
                section = new_section
            para = rest.strip()

            if not para:
                continue

        for piece in _split_long(para, size):
            if buf and len(buf) + len(piece) + 2 > size:
                prev = buf
                flush()
                buf = _tail(prev, overlap)
                buf_section = section

            if not buf:
                buf_section = section
            buf = f"{buf}\n\n{piece}".strip() if buf else piece
            
    flush()
    return chunks
