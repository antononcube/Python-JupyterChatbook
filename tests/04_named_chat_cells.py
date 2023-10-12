import unittest
from LLMPrompts.LLMPrompts import llm_prompt
import IPython.core.interactiveshell as ici


class NamedChatCellsTests(unittest.TestCase):
    def test_01_chat_yoda_cells(self):
        ish = ici.InteractiveShell()
        cells = [
            "%load_ext JupyterChatbook",
            "%%chat -i yoda --prompt=@Yoda" + """
            Hi! Who are you?
            """,
            "%%chat -i yoda" + """
            How many students did you have?
            """,
            "%%chat -i yoda" + """
            !Translate|Russian^
            """
        ]

        for c in cells:
            er = ish.run_cell(c)
            self.assertTrue(er.success)

    def test_02_chat_mad_hatter_cells(self):
        ish = ici.InteractiveShell()
        cells = [
            "%load_ext JupyterChatbook",
            "%%chat_meta -i mh1 --prompt" + """
            @MadHatter
            """,
            "%%chat -i mh1" + """
            Hi! What time it is? #ShortLineIt|80
            """,
            "%%chat -i mh" + """
            Have you seen a girl chasing a rabbit? #HaikuStyled
            """
        ]

        for c in cells:
            er = ish.run_cell(c)
            self.assertTrue(er.success)


if __name__ == '__main__':
    unittest.main()
