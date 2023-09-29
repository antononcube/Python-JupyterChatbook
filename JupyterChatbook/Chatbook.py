from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
from LLMFunctionObjects import llm_synthesize, llm_configuration, llm_evaluator, llm_function, llm_chat
from LLMPrompts import llm_prompt


def unquote(v):
    if isinstance(v, str) and ((v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'"))):
        return v[1:-1]
    return v


@magics_class
class Chatbook(Magics):
    chatObjects = {"NONE": llm_chat('', llm_evaluator='ChatGPT')}

    @magic_arguments()
    @argument('-i', '--chat_id', default="NONE", help="Identifier (name) of the chat object")
    @argument('--conf', type=str, help="LLM service access configuration")
    @argument('--prompt', type=str, help="LLM prompt")
    @argument('--max_tokens', type=int, help="Max number of tokens for the LLM response")
    @argument('--temperature', type=float, help="Temperature to use")
    @argument('--api_key', type=str, help="API key to access the LLM service")
    @cell_magic
    def chat(self, line, cell):
        """
        Chat magic for interacting with LLMs.
        :return: LLM evaluation result.
        """

        args = parse_argstring(self.chat, line)
        args = vars(args)
        args = {k: unquote(v) for k, v in args.items()}
        chatID = args.get("chat_id", "NONE")
        if chatID in self.chatObjects:
            chatObj = self.chatObjects[chatID]
        else:
            args2 = {k: v for k, v in args.items() if k not in ["chat_id", "conf", "prompt"]}
            prompt = args.get("prompt", "")
            chatObj = llm_chat(prompt, llm_evaluator=llm_evaluator(args.get("conf", "ChatGPT"), **args2))
            self.chatObjects[chatID] = chatObj

        # Evaluate the chat message
        res = chatObj.eval(cell.strip())
        new_cell = 'print("{}".format("""' + res + '"""))'

        # Result
        self.shell.run_cell(new_cell)

    @magic_arguments()
    @argument('chat_id', default='all', type=str, help="Identifier (name) of the chat object")
    @cell_magic
    def chat_meta(self, line, cell):
        """
        Chat meta magic for interacting with chat objects.
        :return: LLM evaluation result.
        """

        args = parse_argstring(self.chat_meta, line)
        args = vars(args)
        print(args)
        chatID = args["chat_id"]
        new_cell = ""
        doit = True
        cmd = cell.strip()
        if not (chatID == "all" or chatID in self.chatObjects):
            new_cell = f"Unknown chat object id: {chatID}."
        else:
            if cmd in ["drop", "delete"]:
                if chatID == "all":
                    new_cell = f"Dropped all chat objects {list(self.chatObjects.keys())}."
                    self.chatObjects = {}
                else:
                    del self.chatObjects[chatID]
                    new_cell = f"Dropped the chat object {chatID}."
            elif cmd in ["clear"]:
                if chatID == "all":
                    new_cell = f"The command 'clear' applies to chat objects only."
                else:
                    nm = len(self.chatObjects[chatID].messages)
                    self.chatObjects[chatID].messages = []
                    new_cell = f"Cleared {nm} messages of chat object {chatID}."
            elif cmd == "prompt":
                if chatID == "all":
                    new_cell = f"The command 'prompt' applies to chat objects only."
                else:
                    new_cell = self.chatObjects[chatID].prompt()
            elif cmd in ["print", "say"]:
                if chatID == "all":
                    new_cell = str(self.chatObjects)
                else:
                    self.chatObjects[chatID].print()
                    doit = False

                    # Place result
        if doit:
            self.shell.run_cell('print("{}".format("""' + new_cell + '"""))')
