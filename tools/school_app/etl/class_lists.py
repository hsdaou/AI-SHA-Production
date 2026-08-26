"""
Parse the "<grade> Class List.PDF" reports into the student roster.

Each page carries a caption ("05  A", "K1  A", "12  SA") followed by a column
header row, then one line per student:

    seq | LAST NAME | FIRST NAME | Comp | Parnt | Sibl | Bus Dt | AM | PM | seq

Only the first six matter here. Bus routing is personal data the app has no use
for, so it is not read at all — the cheapest way to avoid leaking a field is to
never load it.

The roster is authoritative for one thing the subject lists are not: which
home section a student belongs to. Subject lists repeat a section letter, but
that column is the group's own view and is truncated on long names; the roster
caption is the school's own grouping. Reconciliation between the two is
reported by build.py rather than silently resolved.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .pdfgrid import Word, column_peaks, pdf_pages, text_of, to_lines

# "05  A", "K1  A", "12  SA", "10  L"
CAPTION_RE = re.compile(r"^(\d{1,2}|K\d)\s+([A-Z]{1,2}\d?)$")
SEQ_RE = re.compile(r"^\d{1,3}$")
NUM_RE = re.compile(r"^\d{3,6}$")
ID_RE = re.compile(r"^\d{4,6}$")


@dataclass
class RosterRow:
    computer_number: str
    last_name: str
    first_name: str
    grade: str          # "K1", "K2", "01".."12"
    section: str        # "A", "SA", "LA", ...
    siblings: int | None
    family_number: str | None
    source: str = ""


@dataclass
class ClassListResult:
    rows: list[RosterRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pages_parsed: int = 0
    pages_skipped: int = 0


def _norm_grade(tok: str) -> str:
    return tok if tok.startswith("K") else f"{int(tok):02d}"


def _id_right_edges(data_lines) -> list[float]:
    """Right-edge positions of the identifier columns, learnt from the data.

    The header row cannot be trusted: `02 Class List.PDF` omits the "Comp" and
    "Parnt" captions altogether, so a header-driven parser reads zero students
    out of an entire grade while reporting success.

    The identifier columns are right-aligned, which makes their LEFT edge move
    with the digit count (a four-digit family number starts 6pt right of a
    five-digit one) but pins their RIGHT edge. Cluster on x1, not x0.
    """
    rows = [[w for w in ln.words if ID_RE.match(w.text) and w.x0 >= 55]
            for ln in data_lines]
    return column_peaks(rows, min_support=0.5,
                        key=lambda w: w.x1) if rows else []


def _split_name(cells: list[Word], peaks: list[float]) -> tuple[str, str]:
    """Split the name region into (last, first) at the given-name column."""
    if len(peaks) >= 2:
        boundary = peaks[1] - 1.0
        last = text_of([w for w in cells if w.x0 < boundary])
        first = text_of([w for w in cells if w.x0 >= boundary])
        return last, first
    # Single column of names: no defensible split, keep it whole.
    return text_of(cells), ""


def parse_class_list(path: str) -> ClassListResult:
    out = ClassListResult()
    source = os.path.basename(path)

    for page_no, words in pdf_pages(path):
        lines = to_lines(words)

        # Captions can appear more than once when a section ends mid-page.
        captions: list[tuple[float, str, str]] = []
        for ln in lines:
            m = CAPTION_RE.match(ln.text.strip())
            if m and ln.words[0].x0 < 120:
                captions.append((ln.y, _norm_grade(m.group(1)), m.group(2)))

        # A data row starts with its sequence number in the left gutter and
        # carries at least two identifier-shaped numbers (own id, family id).
        data_lines = [
            ln for ln in lines
            if SEQ_RE.match(ln.words[0].text) and ln.words[0].x0 <= 55
            and sum(1 for w in ln.words if ID_RE.match(w.text) and w.x0 >= 55) >= 1
        ]
        if not data_lines:
            out.pages_skipped += 1
            continue

        edges = _id_right_edges(data_lines)
        if not edges:
            out.warnings.append(
                f"{source} p{page_no}: {len(data_lines)} candidate rows but no "
                f"identifier column could be located — page skipped")
            out.pages_skipped += 1
            continue
        comp_edge = edges[0]

        if not captions:
            out.warnings.append(
                f"{source} p{page_no}: {len(data_lines)} student rows with no "
                f"section caption — rows dropped rather than guessed")
            out.pages_skipped += 1
            continue

        def comp_word(ln) -> Word | None:
            return next((w for w in ln.words
                         if ID_RE.match(w.text) and abs(w.x1 - comp_edge) <= 3), None)

        keep = [(ln, cw) for ln in data_lines if (cw := comp_word(ln))]
        # Everything left of a row's own identifier is the name; nothing about
        # that depends on how wide the name happened to render.
        name_rows = [[w for w in ln.words[1:] if w.x0 < cw.x0] for ln, cw in keep]
        peaks = column_peaks(name_rows)

        for (ln, cw), name_cells in zip(keep, name_rows):
            grade, section = next(
                ((g, s) for y, g, s in reversed(captions) if y < ln.y),
                (captions[0][1], captions[0][2]))

            tail = [w for w in ln.words if w.x0 > cw.x1]
            parent = next((w.text for w in tail if ID_RE.match(w.text)), None)
            sibl = next((w.text for w in tail
                         if w.text.isdigit() and len(w.text) <= 2
                         and 340 < w.x0 < 410), None)

            last, first = _split_name(name_cells, peaks)
            if not last and not first:
                out.warnings.append(
                    f"{source} p{page_no}: student {cw.text} has no readable name")
            out.rows.append(RosterRow(
                computer_number=cw.text,
                last_name=last,
                first_name=first,
                grade=grade,
                section=section,
                siblings=int(sibl) if sibl else None,
                family_number=parent,
                source=f"{source}#p{page_no}",
            ))
        out.pages_parsed += 1

    return out
