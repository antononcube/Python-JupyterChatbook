import json
import os
import tempfile
import unittest

import IPython.core.interactiveshell as ici

from JupyterChatbook.Chatbook import Chatbook


class AutomaticInitializationTests(unittest.TestCase):

    def setUp(self):
        self._env_init = os.environ.get("PYTHON_CHATBOOK_INIT_FILE")
        self._env_personas = os.environ.get("PYTHON_CHATBOOK_LLM_PERSONAS_CONF")

    def tearDown(self):
        if self._env_init is None:
            os.environ.pop("PYTHON_CHATBOOK_INIT_FILE", None)
        else:
            os.environ["PYTHON_CHATBOOK_INIT_FILE"] = self._env_init

        if self._env_personas is None:
            os.environ.pop("PYTHON_CHATBOOK_LLM_PERSONAS_CONF", None)
        else:
            os.environ["PYTHON_CHATBOOK_LLM_PERSONAS_CONF"] = self._env_personas

    def test_init_and_personas_from_env(self):
        with tempfile.TemporaryDirectory() as td:
            init_path = os.path.join(td, "init.py")
            with open(init_path, "w", encoding="utf-8") as f:
                f.write("CHATBOOK_INIT_VAR = 'ok'\n")

            personas_path = os.path.join(td, "personas.json")
            personas = {
                "test_bot": {
                    "chat_id": "test_bot",
                    "prompt": "You are a helpful assistant.",
                    "conf": "ChatGPT",
                    "model": "gpt-4.1-mini",
                    "temperature": 0.2
                }
            }
            with open(personas_path, "w", encoding="utf-8") as f:
                json.dump(personas, f)

            os.environ["PYTHON_CHATBOOK_INIT_FILE"] = init_path
            os.environ["PYTHON_CHATBOOK_LLM_PERSONAS_CONF"] = personas_path

            ish = ici.InteractiveShell()
            er = ish.run_cell("%load_ext JupyterChatbook")
            self.assertTrue(er.success)
            self.assertEqual(ish.user_ns.get("CHATBOOK_INIT_VAR"), "ok")
            er = ish.run_cell("%%chat_meta --all\nprint\n")
            self.assertTrue(er.success)
            out = er.result
            if hasattr(out, "data"):
                out_str = str(out.data)
            else:
                out_str = str(out)
            self.assertTrue(out_str.strip().startswith("{"))
            self.assertIn("test_bot", out_str)


if __name__ == '__main__':
    unittest.main()
