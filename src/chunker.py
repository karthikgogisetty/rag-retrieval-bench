"""
Section-aware document chunker for the ACME Enterprise document.

Strategy:
- Split by document sections/subsections (detected via heading patterns)
- Target chunk size: 200-400 tokens
- Add 1-sentence overlap between consecutive chunks within a section
- Attach metadata: section_id, section_title, chunk_index
"""

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: int
    text: str
    section_id: str
    section_title: str
    token_count: int  # approximate word count as proxy for tokens

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"Chunk({self.chunk_id}, sec={self.section_id}, ~{self.token_count}tok, '{preview}...')"


# Regex to detect section headings like "5. Enterprise-Grade Security" or "5.1 Data Encryption"
# Handles both "N. Title" (top-level) and "N.N Title" (subsection) formats
HEADING_PATTERN = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2})?)\.?\s+(.+)$", re.MULTILINE
)

# Max length for a line to be considered a heading (filters out numbered list items)
MAX_HEADING_LINE_LENGTH = 80


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word for English text."""
    return int(len(text.split()) * 1.3)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _merge_sentences_into_chunks(
    sentences: list[str],
    section_id: str,
    section_title: str,
    max_tokens: int = 400,
    overlap_sentences: int = 1,
) -> list[dict]:
    """Merge sentences into chunks of approximately max_tokens, with overlap."""
    if not sentences:
        return []

    chunks = []
    current_sentences = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = _estimate_tokens(sentence)

        if current_tokens + sent_tokens > max_tokens and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "text": chunk_text,
                "section_id": section_id,
                "section_title": section_title,
            })
            # Keep last N sentences for overlap
            overlap = current_sentences[-overlap_sentences:]
            current_sentences = overlap
            current_tokens = sum(_estimate_tokens(s) for s in overlap)

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    # Flush remaining
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        # Avoid tiny trailing chunks — merge with previous if too small
        if chunks and _estimate_tokens(chunk_text) < 50:
            chunks[-1]["text"] += " " + chunk_text
        else:
            chunks.append({
                "text": chunk_text,
                "section_id": section_id,
                "section_title": section_title,
            })

    return chunks


def parse_sections(document_text: str) -> list[dict]:
    """Parse document into sections based on heading patterns."""
    lines = document_text.strip().split("\n")
    sections = []
    current_section_id = "0"
    current_section_title = "Preamble"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        match = HEADING_PATTERN.match(stripped)
        # Filter: real headings are short titles, not numbered list items with descriptions
        if match and len(stripped) <= MAX_HEADING_LINE_LENGTH:
            # Save previous section
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append({
                        "section_id": current_section_id,
                        "section_title": current_section_title,
                        "body": body,
                    })
            current_section_id = match.group(1)
            current_section_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({
                "section_id": current_section_id,
                "section_title": current_section_title,
                "body": body,
            })

    return sections


def chunk_document(document_path: str, max_tokens: int = 400) -> list[Chunk]:
    """Main entry point: read document, parse sections, create chunks."""
    with open(document_path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = parse_sections(text)
    all_chunks = []
    chunk_id = 0

    for section in sections:
        sentences = _split_into_sentences(section["body"])
        raw_chunks = _merge_sentences_into_chunks(
            sentences,
            section_id=section["section_id"],
            section_title=section["section_title"],
            max_tokens=max_tokens,
        )
        for raw in raw_chunks:
            all_chunks.append(Chunk(
                chunk_id=chunk_id,
                text=raw["text"],
                section_id=raw["section_id"],
                section_title=raw["section_title"],
                token_count=_estimate_tokens(raw["text"]),
            ))
            chunk_id += 1

    return all_chunks


if __name__ == "__main__":
    import os
    doc_path = os.path.join(os.path.dirname(__file__), "..", "data", "acme_enterprise.txt")
    chunks = chunk_document(doc_path)
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks:
        print(f"  [{c.chunk_id:02d}] sec {c.section_id:<5} ~{c.token_count:3d} tok | {c.section_title}")
    print(f"\nToken stats: min={min(c.token_count for c in chunks)}, "
          f"max={max(c.token_count for c in chunks)}, "
          f"avg={sum(c.token_count for c in chunks)//len(chunks)}")
