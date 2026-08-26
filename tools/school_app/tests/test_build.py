"""Build orchestration and source-provenance safety."""

import hashlib
import os
import zipfile

import pytest

from etl.build import pdf_inputs, source_sha256


def test_zip_reports_are_copied_by_basename_into_private_temp_space(tmp_path):
    archive = tmp_path / "reports.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../outside/01 Class List.PDF", b"first")
        zf.writestr("nested/02 Class List.PDF", b"second")
        zf.writestr("nested/readme.txt", b"ignored")

    with pdf_inputs(str(archive), "Class List") as files:
        assert [os.path.basename(path) for path in files] == [
            "01 Class List.PDF", "02 Class List.PDF"]
        assert all(os.path.exists(path) for path in files)
        assert all(str(tmp_path) not in os.path.realpath(path) for path in files)
        copied = list(files)
    assert all(not os.path.exists(path) for path in copied)


def test_zip_with_duplicate_report_basenames_is_rejected(tmp_path):
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a/01 Class List.PDF", b"first")
        zf.writestr("b/01 Class List.PDF", b"second")
    with pytest.raises(SystemExit, match="duplicate report names"):
        with pdf_inputs(str(archive), "Class List"):
            pass


def test_file_source_hash_is_the_standard_sha256(tmp_path):
    source = tmp_path / "reports.zip"
    source.write_bytes(b"immutable source bytes")
    assert source_sha256(str(source)) == hashlib.sha256(
        source.read_bytes()).hexdigest()
