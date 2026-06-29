#!/usr/bin/env python3
"""Build the AI-SHA school knowledge base from raw Markdown files.

Reads all .md files from ../aisha_raw_data/, splits them by ## headers
into section-aware chunks, embeds with BAAI/bge-small-en-v1.5, and stores
in a ChromaDB persistent collection at ../aisha_knowledge_db/.

The resulting vector store is consumed by admin_node.py at runtime.

Chunk strategy:
  - Each ## header defines a "section" (e.g. "Cambridge IGCSE").
  - Every bullet point (line starting with "- ") under a section becomes
    its own chunk, with the # title and ## section prepended for context.
  - Non-bullet paragraphs are grouped into a single chunk per section.
  - This ensures exam entries are individually retrievable while retaining
    the qualification context (IGCSE vs AS/A Level vs AP).

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

import chromadb


def parse_markdown_sections(filepath: str) -> list[dict]:
    """Parse a Markdown file into section-aware chunks.

    Returns a list of dicts: {text, file_name, section, title}.
    Each bullet point becomes its own chunk; consecutive non-bullet
    lines are merged into paragraph chunks.
    """
    filename = os.path.basename(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = []
    title = ''
    section = ''
    paragraph_lines = []

    def flush_paragraph():
        """Emit accumulated non-bullet lines as a single chunk."""
        if paragraph_lines:
            text = '\n'.join(paragraph_lines).strip()
            if text:
                chunks.append({
                    'text': f"{title} — {section}\n{text}" if section else text,
                    'file_name': filename,
                    'section': section,
                })
            paragraph_lines.clear()

    for line in content.splitlines():
        stripped = line.strip()

        # Top-level title (# header)
        if stripped.startswith('# ') and not stripped.startswith('## '):
            flush_paragraph()
            title = stripped.lstrip('# ').strip()
            continue

        # Section header (## header)
        if stripped.startswith('## '):
            flush_paragraph()
            section = stripped.lstrip('# ').strip()
            continue

        # Bullet point → individual chunk with section context
        if stripped.startswith('- '):
            flush_paragraph()
            entry = stripped.lstrip('- ').strip()
            if entry:
                # Prepend title + section for retrieval context
                context_prefix = f"{title} — {section}" if section else title
                full_text = f"{context_prefix}\n{entry}" if context_prefix else entry
                chunks.append({
                    'text': full_text,
                    'file_name': filename,
                    'section': section,
                })
            continue

        # Regular text → accumulate for paragraph chunk
        if stripped:
            paragraph_lines.append(stripped)

    flush_paragraph()
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

    # Store in ChromaDB
    print(f'Writing to ChromaDB at {output_dir}...')
    os.makedirs(output_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=output_dir)

    # Clear existing collection to avoid stale data
    try:
        client.delete_collection('school_info')
        print('  Cleared existing school_info collection.')
    except ValueError:
        pass  # Collection didn't exist

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
