import unittest
from LLMPrompts.LLMPrompts import llm_prompt
import IPython.core.interactiveshell as ici


class SimpleChatCellsTests(unittest.TestCase):
    def test_03_chat_NONE_cells(self):
        ish = ici.InteractiveShell()
        cells = [
            "%load_ext JupyterChatbook",
            "%%chat" + """
            How many people live in Brazil?
            """,
            "%%chat" + """
            What is the ethnic breakdown?
            """,
            "%%chat" + """
            Where most people live?
            """
        ]

        for c in cells:
            er = ish.run_cell(c)
            self.assertTrue(er.success)

if __name__ == '__main__':
    unittest.main()
