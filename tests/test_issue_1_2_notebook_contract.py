"""Notebook contract checks requiring only Python stdlib and pytest.

Run from repository root after placing the notebook at its required path.
"""
from pathlib import Path
import json

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_eda_and_target_analysis.ipynb"
DOC = ROOT / "docs" / "data_decisions.md"


def test_required_files_are_present():
    assert NOTEBOOK.is_file(), f"Expected issue deliverable: {NOTEBOOK}"
    assert DOC.is_file(), f"Expected issue deliverable: {DOC}"


def test_notebook_json_and_core_sections():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    for token in ("DELAY_SEVERITY", "missing_by_status", "DEP_TIME_BLK",
                  "new_in_2023", "train_end_exclusive", "reference_output",
                  "Arrival Traffic", "Findings"):
        assert token in source, f"Required EDA analysis not found: {token}"


def test_notebook_code_is_parseable():
    import ast
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for idx, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell.get("source", []))
            # Notebook cells can contain Jupyter magic: this notebook is pure Python.
            ast.parse(code, filename=f"cell_{idx}.py")


def test_decision_document_mentions_required_rationale():
    text = DOC.read_text(encoding="utf-8")
    for token in ("ARR_DELAY", "CANCELLED", "DIVERTED", "ORIGIN",
                  "train", "reference", "threshold", "arrival traffic"):
        assert token.lower() in text.lower()
