#!/usr/bin/env python3
"""Build the AI-SHA school knowledge base from raw Markdown files.

Reads all .md files from ../aisha_raw_data/, splits them by ## headers
into section-aware chunks, embeds with BAAI/bge-small-en-v1.5, and stores
in a ChromaDB persistent collection at ../aisha_knowledge_db/.

The resulting vector store is consumed by admin_node.py at runtime.

Chunk strategy:
  - Each ## header defines a "section" (e.g. "Cambridge IGCSE"); deeper
    ###+ headers add a "subsection" folded into each chunk's context.
  - Every list item under a section becomes its own chunk, with the # title
    and ## section prepended for context.  Both unordered markers ("-", "*",
    "+") and ordered markers ("1.", "2)") are recognised.
  - Each Markdown table data row becomes its own chunk, with every cell
    labelled by its column header (e.g. "Grade: KG1 | Annual Fee: 21,000").
  - Non-bullet paragraphs are grouped into a single chunk per section.
  - This ensures exam entries, fee rows and calendar dates are individually
    retrievable while retaining the qualification context.

Metadata stored per chunk:
  - file_name: source .md filename
  - section:   the ## header text (used by admin_node's grade-aware filter)

Usage:
    cd src/aisha_brain/aisha_brain
    python3 build_knowledge.py
    python3 build_knowledge.py --raw-data ../aisha_raw_data --output ../aisha_knowledge_db

Requirements:
    pip install chromadb llama-index-embeddings-huggingface sentence-transformers
"""

import argparse
import os
import re
import sys


# Matches a Markdown list-item marker at the start of a stripped line:
#   "- ", "* ", "+ "   (unordered)   and   "1. ", "2) "   (ordered).
# The required trailing whitespace (\s+) is what keeps horizontal rules
# ("---", "***") and bold spans ("**Note**") from being misread as items.
LIST_ITEM_RE = re.compile(r'^([-*+]|\d+[.)])\s+')


def parse_markdown_sections(filepath: str) -> list[dict]:
    """Parse a Markdown file into section-aware chunks.

    Returns a list of dicts: {text, file_name, section}.  Each list item
    and each table data row becomes its own chunk; consecutive plain-text
    lines are merged into paragraph chunks.  The # title, ## section and
    any deeper (###+) subsection are prepended to every chunk for context.
    """
    filename = os.path.basename(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = []
    title = ''
    section = ''
    subsection = ''
    paragraph_lines = []
    table_lines = []

    def make_prefix() -> str:
        """Join the non-empty title/section/subsection with em-dashes."""
        return ' — '.join(p for p in (title, section, subsection) if p)

    def emit(text: str):
        """Append a chunk, prefixing the current heading context."""
        prefix = make_prefix()
        chunks.append({
            'text': f"{prefix}\n{text}" if prefix else text,
            'file_name': filename,
            'section': section,
        })

    def flush_paragraph():
        """Emit accumulated plain-text lines as a single chunk."""
        if paragraph_lines:
            text = '\n'.join(paragraph_lines).strip()
            if text:
                emit(text)
            paragraph_lines.clear()

    def flush_table():
        """Emit each Markdown table data row as its own chunk.

        The header row labels every cell, so a row reads
        "Col A: x | Col B: y"; the |---| separator row is dropped.
        """
        if not table_lines:
            return
        rows = [
            [c.strip() for c in raw.strip('|').split('|')]
            for raw in table_lines
        ]
        table_lines.clear()
        headers = rows[0]
        data_rows = rows[1:]
        # Drop the |---|:--| separator row that follows the header.
        if data_rows and all(
            cell and set(cell) <= set('-: ') for cell in data_rows[0]
        ):
            data_rows = data_rows[1:]
        for cells in data_rows:
            pairs = [
                f"{headers[i]}: {cell}"
                if i < len(headers) and headers[i] else cell
                for i, cell in enumerate(cells) if cell
            ]
            if pairs:
                emit(' | '.join(pairs))

    for line in content.splitlines():
        stripped = line.strip()

        # Horizontal rule (---, ***, ___) — purely visual, ignore.
        if re.match(r'^([-*_])\1{2,}\s*$', stripped):
            continue

        # Heading (#..######): level sets title / section / subsection.
        header_match = re.match(r'(#{1,6})\s+(.*)', stripped)
        if header_match:
            flush_paragraph()
            flush_table()
            level = len(header_match.group(1))
            text = header_match.group(2).strip()
            if level == 1:
                title, section, subsection = text, '', ''
            elif level == 2:
                section, subsection = text, ''
            else:
                subsection = text
            continue

        # Table row → buffer for row-wise chunking.
        if stripped.startswith('|'):
            flush_paragraph()
            table_lines.append(stripped)
            continue

        # List item (-, *, +, 1., 2)) → individual chunk.
        list_match = LIST_ITEM_RE.match(stripped)
        if list_match:
            flush_paragraph()
            flush_table()
            entry = stripped[list_match.end():].strip()
            if entry:
                emit(entry)
            continue

        # Regular text → accumulate for a paragraph chunk.
        if stripped:
            flush_table()
            paragraph_lines.append(stripped)

    flush_paragraph()
    flush_table()
    return chunks


def build_knowledge_base(raw_data_dir: str, output_dir: str):
    """Build the ChromaDB knowledge base from all .md files in raw_data_dir."""
    # Discover .md files
    md_files = sorted(
        os.path.join(raw_data_dir, f)
        for f in os.listdir(raw_data_dir)
        if f.endswith('.md')
    )
    if not md_files:
        print(f'ERROR: No .md files found in {raw_data_dir}')
        sys.exit(1)

    print(f'Found {len(md_files)} Markdown file(s):')
    for f in md_files:
        print(f'  {f}')

    # Parse all files into chunks
    all_chunks = []
    for filepath in md_files:
        chunks = parse_markdown_sections(filepath)
        print(f'  {os.path.basename(filepath)}: {len(chunks)} chunks')
        all_chunks.extend(chunks)

    if not all_chunks:
        print('ERROR: No chunks extracted from any file.')
        sys.exit(1)

    print(f'\nTotal chunks: {len(all_chunks)}')

    # Load embedding model — MUST match admin_node.py
    print('Loading embedding model (BAAI/bge-small-en-v1.5)...')
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')

    # Embed all chunks
    print('Generating embeddings...')
    texts = [c['text'] for c in all_chunks]
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

    # Store in ChromaDB (imported lazily so the parser above can be used
    # — and unit-tested — without the heavy chromadb dependency installed).
    print(f'Writing to ChromaDB at {output_dir}...')
    import chromadb
    os.makedirs(output_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=output_dir)

    # Clear existing collection to avoid stale data.  Deleting a collection that
    # does not exist raises ValueError on chromadb <1.0 but NotFoundError on
    # chromadb >=1.x, so get_or_create first to guarantee a delete target exists
    # regardless of the installed chromadb version.
    client.get_or_create_collection('school_info')
    client.delete_collection('school_info')
    print('  Reset school_info collection.')

    collection = client.create_collection(
        name='school_info',
        metadata={'hnsw:space': 'cosine'},
    )

    # Add chunks in batches (ChromaDB limit: ~5000 per call)
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        collection.add(
            ids=[f'chunk_{i + j}' for j in range(len(batch))],
            documents=[c['text'] for c in batch],
            metadatas=[{
                'file_name': c['file_name'],
                'section': c['section'],
            } for c in batch],
            embeddings=batch_embeddings,
        )

    print(f'\nDone! {collection.count()} chunks indexed in school_info collection.')
    print(f'Knowledge base path: {os.path.abspath(output_dir)}')
    print('\nTo use: ensure admin_node.py\'s knowledge_db_path parameter points here.')


def main():
    parser = argparse.ArgumentParser(
        description='Build AI-SHA school knowledge base from Markdown files.'
    )
    # Default paths relative to this script's location (src/aisha_brain/aisha_brain/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_raw = os.path.join(script_dir, '..', 'aisha_raw_data')
    default_out = os.path.join(script_dir, '..', 'aisha_knowledge_db')

    parser.add_argument(
        '--raw-data', default=default_raw,
        help=f'Directory containing .md source files (default: {default_raw})',
    )
    parser.add_argument(
        '--output', default=default_out,
        help=f'ChromaDB output directory (default: {default_out})',
    )
    args = parser.parse_args()

    raw_data_dir = os.path.abspath(args.raw_data)
    output_dir = os.path.abspath(args.output)

    if not os.path.isdir(raw_data_dir):
        print(f'ERROR: Raw data directory not found: {raw_data_dir}')
        sys.exit(1)

    print('AI-SHA Knowledge Base Builder')
    print(f'  Raw data:  {raw_data_dir}')
    print(f'  Output:    {output_dir}')
    print()

    build_knowledge_base(raw_data_dir, output_dir)


if __name__ == '__main__':
    main()
