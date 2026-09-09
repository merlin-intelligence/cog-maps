"""Chunking — ChunkNorris based: parse every supported format to markdown, then chunk.

Each format becomes a markdown document and is split by the :class:`MarkdownChunker`,
so chunk granularity stays homogeneous across the corpus regardless of source format.
"""
from __future__ import annotations

import logging
import os

from chunknorris.chunkers import MarkdownChunker
from chunknorris.parsers import CSVParser, DocxParser, ExcelParser, MarkdownParser, PdfParser
from collections import Counter
from chunknorris.parsers.pdf.pdf_parser import TextLine

from cogmaps.config import get_ocr_languages, ocr_available

logger = logging.getLogger(__name__)


@staticmethod
def _safe_get_line_spacing(lines: list[TextLine]) -> float:
    linespace_counts = Counter(
        (
            round(curr_line.bbox.y0 - prev_line.bbox.y1, 1)
            for curr_line, prev_line in zip(lines[1:], lines[:-1])
        )
    )
    if not linespace_counts:
        return 0.0
    return max(linespace_counts, key=linespace_counts.get)


# Monkeypatch PdfParser._get_line_spacing to handle empty linespace_counts
# (upstream bug: ValueError: max() arg is an empty sequence). Guarded so that
# a future chunknorris release renaming/removing this private method degrades
# to a warning (PDF ingestion may hit the original bug again) instead of
# crashing the whole app at import time.
if hasattr(PdfParser, "_get_line_spacing"):
    PdfParser._get_line_spacing = _safe_get_line_spacing
else:
    logger.warning(
        "chunknorris.PdfParser has no _get_line_spacing to patch — "
        "the empty-linespace-counts fix is not applied; upgrade or pin chunknorris."
    )


def _file_parser_for(ext: str):
    """Return a ChunkNorris parser that reads ``ext`` files directly, or None.

    PPTX and TXT are excluded here — they are turned into markdown strings first and
    parsed via :meth:`MarkdownParser.parse_string` in :func:`chunk_with_chunknorris`.
    """
    if ext == ".pdf":
        return PdfParser(
            use_ocr="auto" if ocr_available() else "never",
            ocr_language=get_ocr_languages(),
        )
    if ext == ".docx":
        return DocxParser()
    if ext == ".xlsx":
        return ExcelParser()
    if ext == ".csv":
        # Delimiter is auto-detected; rows are emitted as JSON lines.
        return CSVParser()
    if ext == ".md":
        return MarkdownParser()
    return None


def _pptx_to_markdown(filepath: str) -> str:
    """Convert a .pptx deck to markdown via MarkItDown (no LLM image captioning)."""
    from markitdown import MarkItDown

    return MarkItDown().convert(filepath).text_content


def chunk_with_chunknorris(filepath: str):
    """Parse a supported file to markdown via ChunkNorris and return its chunks.

    PDF, DOCX, XLSX, CSV and MD are parsed natively by ChunkNorris. TXT is read as raw
    markdown text, and PPTX is first converted to markdown with MarkItDown. Every path
    ends in :class:`MarkdownParser` + :class:`MarkdownChunker`. Raises :class:`ValueError`
    for any other extension.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pptx":
        doc = MarkdownParser().parse_string(_pptx_to_markdown(filepath))
    elif ext in (".txt", ".md"):
        # MarkdownParser.parse_file only accepts ".md" but can be strict with encoding.
        # We read manually with errors="replace" (not "ignore") so a non-UTF8 file
        # still ingests, with bad bytes visibly turned into '�' rather than
        # silently dropped.
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if "�" in text:
            logger.warning(
                "%s is not valid UTF-8 — some characters were replaced with '�'.", filepath
            )
        doc = MarkdownParser().parse_string(text)
    else:
        parser = _file_parser_for(ext)
        if parser is None:
            raise ValueError(f"format '{ext}' is not supported by ChunkNorris")
        doc = parser.parse_file(filepath)

    return MarkdownChunker().chunk(doc)
