"""
Geometry-first extraction for Microsoft Access report PDFs.

WHY NOT `pdftotext -layout`
--------------------------
The class/subject list reports use fixed-width report fields. When a value fills
its field, `-layout` has no whitespace left to separate it from the next column
and silently welds them together:

    05 Class List, row 28:   "UMAIR NAVEED MUHAMMAD EZA        23075"
                                             ^^^^^^^^^^^^^^^^
                             first name "MUHAMMAD EZ" + section "A"

A parser splitting that row on whitespace invents a student called "EZA" in no
section at all. `-bbox-layout` gives every word an x/y box, so the columns can be
recovered from geometry, which does not depend on whether a value happened to be
short enough that day.

The two report families differ in layout but share this shape: a handful of
header lines, then numbered data rows whose columns sit at stable x positions.
`column_peaks` recovers those positions from the data itself rather than from
hard-coded offsets, so a report that shifts by a few points still parses.
"""

from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class Line:
    y: float
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def between(self, x_lo: float, x_hi: float) -> list[Word]:
        return [w for w in self.words if x_lo <= w.x0 < x_hi]


# A bare `&` inside a subject name ("Moral, Soc. & Cultural Stud.") is emitted
# raw by the report writer and is not valid XML.
_BARE_AMP = re.compile(r"&(?!(amp|lt|gt|quot|apos|#\d+);)")


def pdf_pages(path: str, first: int | None = None, last: int | None = None):
    """Yield (page_number, [Word]) for each page, using word bounding boxes."""
    cmd = ["pdftotext", "-bbox-layout"]
    if first is not None:
        cmd += ["-f", str(first)]
    if last is not None:
        cmd += ["-l", str(last)]
    cmd += [path, "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"pdftotext failed on {path}: {proc.stderr.strip()[:300]}")
    root = ET.fromstring(_BARE_AMP.sub("&amp;", proc.stdout))

    page_no = (first or 1) - 1
    for page in (el for el in root.iter() if el.tag.endswith("page")):
        page_no += 1
        words = [
            Word(el.text.strip(), float(el.get("xMin")), float(el.get("xMax")),
                 float(el.get("yMin")), float(el.get("yMax")))
            for el in page.iter()
            if el.tag.endswith("word") and el.text and el.text.strip()
        ]
        yield page_no, words


def to_lines(words: list[Word], tol: float = 1.6) -> list[Line]:
    """Cluster words into visual lines by baseline, then order each line by x.

    `tol` is deliberately tight. These reports place a row's e-mail 0.4pt above
    the rest of the row (different font metrics) — that must merge — while the
    section caption sits only 2.0pt above the column headers — that must not.
    A tolerance of 3pt, as used by the previous implementation, merges the two
    and the caption is then read as part of the header row.
    """
    lines: list[Line] = []
    for w in sorted(words, key=lambda w: (w.y0, w.x0)):
        for ln in lines:
            if abs(ln.y - w.y0) <= tol:
                ln.words.append(w)
                break
        else:
            lines.append(Line(w.y0, [w]))
    for ln in lines:
        ln.words.sort(key=lambda w: w.x0)
    return sorted(lines, key=lambda ln: ln.y)


def column_peaks(rows: list[list[Word]], min_support: float = 0.6,
                 quantum: float = 1.0, key=lambda w: w.x0) -> list[float]:
    """Recover column edge positions from a page's data rows.

    Every data row contributes exactly one word at each column's edge, so those
    x values pile up into a sharp peak. Continuation words ("MARIA" as a second
    given name) scatter and fall below `min_support`.

    Pass `key=lambda w: w.x1` for right-aligned columns, where the left edge
    moves with the value's width but the right edge does not.

    Returns the supported positions in left-to-right order.
    """
    if not rows:
        return []
    hist: Counter[float] = Counter()
    for row in rows:
        for x in {round(key(w) / quantum) * quantum for w in row}:
            hist[x] += 1
    threshold = max(2, int(len(rows) * min_support))
    peaks = sorted(x for x, n in hist.items() if n >= threshold)

    # Anti-aliasing: two adjacent quanta can both clear the threshold for one
    # physical column. Keep the stronger of any pair closer than 4pt.
    merged: list[float] = []
    for x in peaks:
        if merged and x - merged[-1] < 4.0:
            if hist[x] > hist[merged[-1]]:
                merged[-1] = x
        else:
            merged.append(x)
    return merged


def split_at(words: list[Word], boundaries: list[float]) -> list[list[Word]]:
    """Bucket words into len(boundaries) cells by left edge.

    A word belongs to the last column whose start is <= its own start, with a
    1pt slack so a glyph that renders a hair left of the column rule does not
    fall into the previous cell.
    """
    cells: list[list[Word]] = [[] for _ in boundaries]
    for w in words:
        idx = None
        for i, b in enumerate(boundaries):
            if w.x0 >= b - 1.0:
                idx = i
            else:
                break
        if idx is not None:
            cells[idx].append(w)
    return cells


def text_of(cell: list[Word]) -> str:
    return re.sub(r"\s+", " ", " ".join(w.text for w in cell)).strip()
