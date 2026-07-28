import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "01_Q_Fragestellung.ipynb",
    ROOT / "notebooks" / "02_U_Datenverstaendnis.ipynb",
    ROOT / "notebooks" / "03_A3_Algorithmen_Features_Hyperparameter.ipynb",
    ROOT / "notebooks" / "04_C_Schlussfolgern_und_Vergleichen.ipynb",
    ROOT / "notebooks" / "05_K_Wissenstransfer.ipynb",
]


class QuackNotebookTest(unittest.TestCase):
    def test_each_phase_has_a_valid_project_notebook(self):
        self.assertEqual(len(NOTEBOOKS), 5)
        for path in NOTEBOOKS:
            with self.subTest(path=path.name):
                self.assertTrue(path.exists())
                content = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(content["nbformat"], 4)

                markdown_text = "\n".join(
                    cell["source"]
                    for cell in content["cells"]
                    if cell["cell_type"] == "markdown"
                )
                self.assertIn("Umsetzung im Projekt", markdown_text)

                code_cells = [
                    cell["source"]
                    for cell in content["cells"]
                    if cell["cell_type"] == "code"
                ]
                self.assertGreater(len(code_cells), 0)
                for source in code_cells:
                    ast.parse(source)


if __name__ == "__main__":
    unittest.main()
