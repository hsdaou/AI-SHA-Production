"""Unit tests for build_knowledge.parse_markdown_sections.

These lock in the Markdown chunking rules the school knowledge base
relies on: list-marker variants, ###+ subsections, table rows and
horizontal-rule handling.  The parser imports chromadb lazily, so these
run without a vector store or any heavy dependency.
"""

from aisha_brain.build_knowledge import parse_markdown_sections


def _parse(tmp_path, text):
    """Write *text* to a temp .md file and return its parsed chunks."""
    md = tmp_path / 'sample.md'
    md.write_text(text, encoding='utf-8')
    return parse_markdown_sections(str(md))


def _last_lines(chunks):
    """Return the final (content) line of every chunk's text."""
    return [c['text'].splitlines()[-1] for c in chunks]


class TestListMarkers:

    def test_all_unordered_markers_split(self, tmp_path):
        chunks = _parse(tmp_path, '# T\n## S\n- dash\n* star\n+ plus\n')
        assert _last_lines(chunks) == ['dash', 'star', 'plus']

    def test_ordered_markers_split_and_stripped(self, tmp_path):
        chunks = _parse(tmp_path, '# T\n## S\n1. first\n2) second\n')
        assert _last_lines(chunks) == ['first', 'second']

    def test_item_carries_title_and_section(self, tmp_path):
        chunks = _parse(tmp_path, '# T\n## Cambridge IGCSE\n- French\n')
        assert chunks[0]['section'] == 'Cambridge IGCSE'
        assert chunks[0]['text'] == 'T — Cambridge IGCSE\nFrench'


class TestSubsections:

    def test_h3_folded_into_context_not_leaked(self, tmp_path):
        text = '# Cal\n## Term 1\n### Key Dates\n- Nov 10: exams\n'
        chunks = _parse(tmp_path, text)
        assert '#' not in chunks[0]['text']
        assert chunks[0]['text'] == 'Cal — Term 1 — Key Dates\nNov 10: exams'
        # section metadata stays the ## value — the grade filter needs it.
        assert chunks[0]['section'] == 'Term 1'

    def test_h3_resets_on_new_h2(self, tmp_path):
        text = '# Cal\n## Term 1\n### Start\n- a\n## Term 2\n- b\n'
        chunks = _parse(tmp_path, text)
        assert chunks[0]['text'] == 'Cal — Term 1 — Start\na'
        assert chunks[1]['text'] == 'Cal — Term 2\nb'


class TestTables:

    def test_each_data_row_is_one_labelled_chunk(self, tmp_path):
        text = (
            '# Fees\n## Tuition\n'
            '| Grade | Annual Fee |\n'
            '|-------|-----------|\n'
            '| KG1 | 21,000 |\n'
            '| Grade 7 | 31,500 |\n'
        )
        chunks = _parse(tmp_path, text)
        assert _last_lines(chunks) == [
            'Grade: KG1 | Annual Fee: 21,000',
            'Grade: Grade 7 | Annual Fee: 31,500',
        ]
        assert all(c['section'] == 'Tuition' for c in chunks)

    def test_separator_row_dropped(self, tmp_path):
        text = '# Fees\n## Tuition\n| A | B |\n|---|---|\n| 1 | 2 |\n'
        chunks = _parse(tmp_path, text)
        assert len(chunks) == 1
        assert '---' not in chunks[0]['text']


class TestHorizontalRules:

    def test_rules_produce_no_chunks(self, tmp_path):
        chunks = _parse(tmp_path, '# T\n## S\n- a\n\n---\n\n***\n\n- b\n')
        assert _last_lines(chunks) == ['a', 'b']
