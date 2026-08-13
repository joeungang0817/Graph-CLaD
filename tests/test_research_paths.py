import tempfile
import unittest
from pathlib import Path

from scripts.research_paths import resolve_research_paths


class ResearchPathsTest(unittest.TestCase):
    def test_explicit_local_roots_do_not_require_colab(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = resolve_research_paths(
                project_root=root,
                artifact_root=root / "artifacts",
                libero_root=root / "LIBERO",
                environ={},
            )
            self.assertFalse(paths.colab)
            self.assertEqual(paths.project_root, root)
            self.assertEqual(paths.output_root, root / "outputs")
            self.assertEqual(paths.artifact_root, root / "artifacts")

    def test_environment_overrides_are_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = resolve_research_paths(
                environ={
                    "GRAPH_CLAD_PROJECT_ROOT": str(root / "repo"),
                    "GRAPH_CLAD_ARTIFACT_ROOT": str(root / "artifacts"),
                    "GRAPH_CLAD_LIBERO_ROOT": str(root / "libero"),
                }
            )
            self.assertEqual(paths.project_root, root / "repo")
            self.assertEqual(paths.artifact_root, root / "artifacts")
            self.assertEqual(paths.libero_root, root / "libero")


if __name__ == "__main__":
    unittest.main()
