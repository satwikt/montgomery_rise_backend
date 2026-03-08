"""
ingestion.py — Parse and chunk RISE knowledge-base ``.txt`` files.

File format
-----------
Each file contains one or more document blocks separated by ``---`` lines.
Every block begins with a short header section::

    DOCUMENT: <human-readable title>
    PARCEL_ID: <A | B | C | general>
    TOPIC: <snake_case topic name>

    <free-form content text>

This module extracts those metadata fields and returns clean ``Chunk``
objects ready to be embedded and stored in ChromaDB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .config import DATA_DIR, MAX_CHUNK_SIZE

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

_HEADER_FIELDS: frozenset[str] = frozenset({"DOCUMENT", "PARCEL_ID", "TOPIC"})
_SEPARATOR_PATTERN: re.Pattern[str] = re.compile(r"(?m)^---\s*$")


@dataclass
class Chunk:
    """A single parsed document block with associated ChromaDB metadata."""

    chunk_id: str         # Unique key for ChromaDB  (e.g. ``parcel_a__003``)
    text: str             # Full text content (header prefix + body)
    source_file: str      # Stem of the source ``.txt`` file
    document_title: str   # Value of the ``DOCUMENT:`` header field
    parcel_id: str        # Value of the ``PARCEL_ID:`` header field
    topic: str            # Value of the ``TOPIC:`` header field
    metadata: dict[str, str | int] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Private parsing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _extract_header_field(lines: list[str], field_name: str) -> str:
    """Return the value of *field_name* from *lines*, or ``'unknown'``."""
    prefix = f"{field_name}:"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return "unknown"


def _split_into_raw_blocks(content: str) -> list[str]:
    """Split *content* on ``---`` separator lines; discard empty blocks."""
    return [b.strip() for b in _SEPARATOR_PATTERN.split(content) if b.strip()]


def _parse_block(block: str, source_file: str, block_index: int) -> Chunk | None:
    """
    Parse a single raw document block into a ``Chunk``.

    The parser walks line-by-line, collecting header lines until it hits a
    blank line (the header/body separator) or a non-header line (body starts
    immediately).

    Returns ``None`` for blocks whose body text is too short to be useful.
    """
    lines = block.splitlines()
    header_lines: list[str] = []
    content_lines: list[str] = []
    in_header = True

    for line in lines:
        if in_header:
            stripped = line.strip()
            is_header_line = any(
                stripped.startswith(f"{fn}:") for fn in _HEADER_FIELDS
            )
            if is_header_line:
                header_lines.append(line)
            elif stripped == "" and header_lines:
                # Blank line after at least one header → transition to body.
                in_header = False
            elif stripped == "":
                pass  # Leading blank before any header — skip.
            else:
                # Non-blank non-header line with no prior header → body-only block.
                in_header = False
                content_lines.append(line)
        else:
            content_lines.append(line)

    document_title = _extract_header_field(header_lines, "DOCUMENT")
    parcel_id = _extract_header_field(header_lines, "PARCEL_ID")
    topic = _extract_header_field(header_lines, "TOPIC")

    body = "\n".join(content_lines).strip()
    if len(body) < 50:
        return None  # Too short to be useful — skip.

    # Prefix the body with the document title so the LLM has
    # immediate context even when reading a single chunk in isolation.
    text = f"[{document_title}]\n\n{body}" if document_title != "unknown" else body

    return Chunk(
        chunk_id=f"{source_file}__{block_index:03d}",
        text=text,
        source_file=source_file,
        document_title=document_title,
        parcel_id=parcel_id,
        topic=topic,
        metadata={
            "source_file": source_file,
            "document_title": document_title,
            "parcel_id": parcel_id,
            "topic": topic,
            "block_index": block_index,
        },
    )


def _split_long_chunk(chunk: Chunk, max_size: int = MAX_CHUNK_SIZE) -> list[Chunk]:
    """
    Split *chunk* on paragraph boundaries if its text exceeds *max_size*.

    Sub-chunks inherit all metadata from the parent and receive an
    additional ``sub_index`` field.  If the chunk is within the size limit
    it is returned unchanged inside a one-element list.
    """
    if len(chunk.text) <= max_size:
        return [chunk]

    paragraphs = chunk.text.split("\n\n")
    sub_chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_len = 0
    sub_index = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 accounts for the "\n\n" separator
        if current_len + para_len > max_size and current_parts:
            sub_chunks.append(
                Chunk(
                    chunk_id=f"{chunk.chunk_id}_sub{sub_index:02d}",
                    text="\n\n".join(current_parts),
                    source_file=chunk.source_file,
                    document_title=chunk.document_title,
                    parcel_id=chunk.parcel_id,
                    topic=chunk.topic,
                    metadata={**chunk.metadata, "sub_index": sub_index},
                )
            )
            current_parts = [para]
            current_len = para_len
            sub_index += 1
        else:
            current_parts.append(para)
            current_len += para_len

    if current_parts:
        sub_chunks.append(
            Chunk(
                chunk_id=f"{chunk.chunk_id}_sub{sub_index:02d}",
                text="\n\n".join(current_parts),
                source_file=chunk.source_file,
                document_title=chunk.document_title,
                parcel_id=chunk.parcel_id,
                topic=chunk.topic,
                metadata={**chunk.metadata, "sub_index": sub_index},
            )
        )

    return sub_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def parse_file(filepath: Path) -> list[Chunk]:
    """
    Parse a single knowledge-base ``.txt`` file into a list of ``Chunk`` objects.

    Parameters
    ----------
    filepath:
        Absolute path to the ``.txt`` file.

    Returns
    -------
    list[Chunk]
        All chunks extracted from the file, with oversized blocks already
        split on paragraph boundaries.
    """
    content = filepath.read_text(encoding="utf-8")
    source_file = filepath.stem  # e.g. "parcel_a_heritage"

    chunks: list[Chunk] = []
    for i, block in enumerate(_split_into_raw_blocks(content)):
        chunk = _parse_block(block, source_file, i)
        if chunk is not None:
            chunks.extend(_split_long_chunk(chunk))

    return chunks


def load_all_chunks(data_dir: Path = DATA_DIR) -> list[Chunk]:
    """
    Load and parse every ``.txt`` file in *data_dir*.

    Parameters
    ----------
    data_dir:
        Directory that contains the knowledge-base ``.txt`` files.
        Defaults to ``config.DATA_DIR``.

    Returns
    -------
    list[Chunk]
        Flat list of all chunks across all files, sorted by file name.

    Raises
    ------
    FileNotFoundError
        If no ``.txt`` files are found in *data_dir*.
    """
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir}. "
            "Copy your knowledge-base files into the data/ folder."
        )

    all_chunks: list[Chunk] = []
    for filepath in txt_files:
        all_chunks.extend(parse_file(filepath))
    return all_chunks


def iter_chunks(data_dir: Path = DATA_DIR) -> Iterator[Chunk]:
    """
    Memory-efficient generator version of :func:`load_all_chunks`.

    Yields one ``Chunk`` at a time instead of collecting them all in memory.
    """
    for filepath in sorted(data_dir.glob("*.txt")):
        yield from parse_file(filepath)
