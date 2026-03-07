"""
ingestion.py — Parse and chunk RISE knowledge base .txt files.

Each file contains multiple documents separated by '---' lines.
Each document block has a header section with metadata fields:
    DOCUMENT: <title>
    PARCEL_ID: <id>
    TOPIC: <topic>

This module extracts those metadata fields and returns clean Chunk objects
ready to be embedded and stored in ChromaDB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .config import DATA_DIR, DOCUMENT_SEPARATOR, MAX_CHUNK_SIZE


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single document chunk with metadata."""
    chunk_id: str           # Unique ID for ChromaDB (filename_chunkindex)
    text: str               # The actual content text
    source_file: str        # Which .txt file this came from
    document_title: str     # The DOCUMENT: header value
    parcel_id: str          # PARCEL_ID: value (e.g., "A", "B", "C", "general")
    topic: str              # TOPIC: value (e.g., "scores", "grants")
    metadata: dict = field(default_factory=dict)  # Extra metadata for ChromaDB


# ─── Parsing Helpers ──────────────────────────────────────────────────────────

def _extract_header_field(lines: list[str], field_name: str) -> str:
    """Extract a metadata field value from header lines."""
    prefix = f"{field_name}:"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return "unknown"


def _split_into_raw_blocks(content: str) -> list[str]:
    """
    Split file content on '---' separator lines into raw text blocks.
    Filters out empty blocks.
    """
    # Split on lines that are exactly '---' (with optional surrounding whitespace)
    blocks = re.split(r'(?m)^---\s*$', content)
    return [b.strip() for b in blocks if b.strip()]


def _parse_block(block: str, source_file: str, block_index: int) -> Chunk | None:
    """
    Parse a single raw document block into a Chunk.
    
    A block looks like:
        DOCUMENT: Montgomery Alabama — Community and Economic Context
        PARCEL_ID: general
        TOPIC: community_overview
        
        <actual content text>
    
    Returns None if the block is too short to be useful.
    """
    lines = block.splitlines()
    
    # Detect where the header ends and content begins.
    # Header lines start with known field names; content starts after a blank line
    # OR after all header lines are consumed.
    header_field_names = {"DOCUMENT", "PARCEL_ID", "TOPIC"}
    header_lines: list[str] = []
    content_lines: list[str] = []
    in_header = True

    for line in lines:
        if in_header:
            stripped = line.strip()
            # Check if this looks like a header field
            is_header_line = any(
                stripped.startswith(f"{fn}:") for fn in header_field_names
            )
            if is_header_line:
                header_lines.append(line)
            elif stripped == "":
                # Blank line after header fields = transition to content
                if header_lines:
                    in_header = False
                # else: leading blank lines before any header, skip
            else:
                # Non-blank, non-header line with no header yet = content-only block
                in_header = False
                content_lines.append(line)
        else:
            content_lines.append(line)

    # Extract metadata
    document_title = _extract_header_field(header_lines, "DOCUMENT")
    parcel_id = _extract_header_field(header_lines, "PARCEL_ID")
    topic = _extract_header_field(header_lines, "TOPIC")

    content_text = "\n".join(content_lines).strip()

    # Skip trivially short blocks
    if len(content_text) < 50:
        return None

    # Build a descriptive prefix to help the LLM understand context
    if document_title != "unknown":
        prefixed_text = f"[{document_title}]\n\n{content_text}"
    else:
        prefixed_text = content_text

    chunk_id = f"{source_file}__{block_index:03d}"

    return Chunk(
        chunk_id=chunk_id,
        text=prefixed_text,
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
    If a chunk's text exceeds max_size characters, split it on paragraph
    boundaries into sub-chunks. Each sub-chunk inherits the parent metadata.
    """
    if len(chunk.text) <= max_size:
        return [chunk]

    paragraphs = chunk.text.split("\n\n")
    sub_chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_len = 0
    sub_index = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 for the \n\n
        if current_len + para_len > max_size and current_parts:
            sub_text = "\n\n".join(current_parts)
            sub_chunks.append(Chunk(
                chunk_id=f"{chunk.chunk_id}_sub{sub_index:02d}",
                text=sub_text,
                source_file=chunk.source_file,
                document_title=chunk.document_title,
                parcel_id=chunk.parcel_id,
                topic=chunk.topic,
                metadata={**chunk.metadata, "sub_index": sub_index},
            ))
            current_parts = [para]
            current_len = para_len
            sub_index += 1
        else:
            current_parts.append(para)
            current_len += para_len

    # Flush remaining
    if current_parts:
        sub_text = "\n\n".join(current_parts)
        sub_chunks.append(Chunk(
            chunk_id=f"{chunk.chunk_id}_sub{sub_index:02d}",
            text=sub_text,
            source_file=chunk.source_file,
            document_title=chunk.document_title,
            parcel_id=chunk.parcel_id,
            topic=chunk.topic,
            metadata={**chunk.metadata, "sub_index": sub_index},
        ))

    return sub_chunks


# ─── Public API ───────────────────────────────────────────────────────────────

def parse_file(filepath: Path) -> list[Chunk]:
    """
    Parse a single knowledge base .txt file into a list of Chunks.
    """
    content = filepath.read_text(encoding="utf-8")
    source_file = filepath.stem  # e.g., "parcel_a_heritage"

    raw_blocks = _split_into_raw_blocks(content)
    chunks: list[Chunk] = []

    for i, block in enumerate(raw_blocks):
        chunk = _parse_block(block, source_file, i)
        if chunk is None:
            continue
        # Split any oversized chunks
        sub_chunks = _split_long_chunk(chunk)
        chunks.extend(sub_chunks)

    return chunks


def load_all_chunks(data_dir: Path = DATA_DIR) -> list[Chunk]:
    """
    Load and parse all .txt files in the data directory.
    Returns a flat list of all Chunks across all files.
    """
    txt_files = sorted(data_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir}. "
            "Please copy your knowledge base files into the data/ folder."
        )

    all_chunks: list[Chunk] = []
    for filepath in txt_files:
        file_chunks = parse_file(filepath)
        all_chunks.extend(file_chunks)

    return all_chunks


def iter_chunks(data_dir: Path = DATA_DIR) -> Iterator[Chunk]:
    """Generator version of load_all_chunks for memory-efficient processing."""
    txt_files = sorted(data_dir.glob("*.txt"))
    for filepath in txt_files:
        yield from parse_file(filepath)
