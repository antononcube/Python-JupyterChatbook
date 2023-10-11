import unittest
from LLMPrompts.LLMPrompts import llm_prompt
import IPython.core.interactiveshell as ici


class PaLMCellsTests(unittest.TestCase):

    def test_01_palm_cells(self):
        ish = ici.InteractiveShell()
        cells = [
            "%load_ext JupyterChatbook",
            "%%palm" + """
            How many people live in Portugal?
            """,
            "%%palm" + """
            Write a funny story in Russian
            """
        ]

        for c in cells:
            er = ish.run_cell(c)
            self.assertTrue(er.success)

    def test_02_palm_cells(self):
        ish = ici.InteractiveShell()
        cells = [
            "%load_ext JupyterChatbook",
            "%%palm" + """
            Write a funny story in Russian.
            """
        ]

        for c in cells:
            er = ish.run_cell(c)
            self.assertTrue(er.success)


if __name__ == '__main__':
    unittest.main()
