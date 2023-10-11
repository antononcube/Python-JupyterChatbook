import unittest
from LLMPrompts.LLMPrompts import llm_prompt
import IPython.core.interactiveshell as ici


class OpenAIMagicCellsTests(unittest.TestCase):

    def test_01_openai_cells1(self):
        ish = ici.InteractiveShell()
        cells = [
            "%load_ext JupyterChatbook",
            "%%openai" + """
            How many people live in Italy?
            """,
            "%%chatgpt --temperature 1.3" + """
            Write a haiku for autumn dawn.
            """
        ]

        for c in cells:
            er = ish.run_cell(c)
            self.assertTrue(er.success)

    def test_02_dalle(self):
        ish = ici.InteractiveShell()
        cells = [
            "%load_ext JupyterChatbook",
            "%%dalle --size=small --copy_to_clipboard=" + """
            Tundra landscape painting in the style of Rafael.
            """
        ]

        for c in cells:
            er = ish.run_cell(c)
            self.assertTrue(er.success)


if __name__ == '__main__':
    unittest.main()
