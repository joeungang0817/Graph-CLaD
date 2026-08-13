import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORDERED_NOTEBOOKS = [
    "00_environment_and_paths.ipynb",
    "phase_0_clad_baseline_smoke.ipynb",
    "phase_1a_state_api_audit.ipynb",
    "phase_2a_static_graph_contract.ipynb",
    "phase_2r_scripted_diagnostics.ipynb",
    "phase_2d_official_demo_dataset.ipynb",
    "phase_3a_dataset_and_label_qa.ipynb",
    "phase_3b_corrected_architecture_gate.ipynb",
    "phase_3b_evaluation_and_controls.ipynb",
]
OBSOLETE_SEQUENCE_NAMES = [
    "phase_00_environment_and_paths.ipynb",
    "phase_01_baseline_and_state_contracts.ipynb",
    "phase_02_graph_dataset_preparation.ipynb",
    "phase_03_corrected_manifest.ipynb",
    "phase_04_architecture_gate_training.ipynb",
    "phase_05_evaluation_and_weak_label_qa.ipynb",
]


class NotebookStructureTest(unittest.TestCase):
    def test_official_phase_names_replace_temporary_sequence_names(self):
        for name in ORDERED_NOTEBOOKS:
            self.assertTrue((ROOT / "notebooks" / name).is_file(), name)
        for name in OBSOLETE_SEQUENCE_NAMES:
            self.assertFalse((ROOT / "notebooks" / name).exists(), name)

    def test_all_notebooks_are_valid_v4_json(self):
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            with self.subTest(path=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)
                self.assertIsInstance(notebook["cells"], list)

    def test_ordered_notebooks_have_required_runbook_sections(self):
        for name in ORDERED_NOTEBOOKS:
            path = ROOT / "notebooks" / name
            with self.subTest(path=name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                text = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in notebook["cells"]
                )
                for required in ("목적", "입력", "출력", "다음"):
                    self.assertIn(required, text)
                self.assertTrue(
                    any(cell["cell_type"] == "code" for cell in notebook["cells"])
                )
                self.assertEqual(
                    sum(len(cell.get("outputs", [])) for cell in notebook["cells"]),
                    0,
                )

    def test_ordered_notebook_code_cells_parse_as_python(self):
        for name in ORDERED_NOTEBOOKS:
            path = ROOT / "notebooks" / name
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                with self.subTest(path=name, cell=index):
                    ast.parse("".join(cell.get("source", [])))


if __name__ == "__main__":
    unittest.main()
