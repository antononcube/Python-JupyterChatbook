from typing import Union

from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
from LLMFunctionObjects import llm_configuration, llm_evaluator, llm_chat
from LLMPrompts import llm_prompt_expand
import openai
import google.generativeai
import os
import pyperclip
import IPython
from IPython import display
from base64 import b64decode


def _unquote(v):
    if isinstance(v, str) and ((v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'"))):
        return v[1:-1]
    return v


def _prep_display(cell, fmt="asis"):
    new_cell = cell.replace('"""', '\\\"\\\"\\\"')
    if fmt == "html":
        new_cell = "import IPython\nIPython.display.HTML(" + '"{}".format("""' + new_cell + '"""))'
    elif fmt in ["markdown", "md"]:
        new_cell = "import IPython\nIPython.display.display_markdown(" + '"{}".format("""' + new_cell + '"""), raw=True)'
    else:
        new_cell = 'print("{}".format("""' + new_cell + '"""))'
    return new_cell


def _prep_result(cell, fmt="pretty"):
    new_cell = cell
    if not isinstance(new_cell, str):
        new_cell = repr(new_cell)
    if fmt == "html":
        new_cell = IPython.display.HTML(new_cell)
    elif fmt in ["markdown", "md"]:
        new_cell = IPython.display.display_markdown(new_cell, raw=True)
    elif fmt == "pretty":
        new_cell = IPython.display.Pretty(new_cell)
    return new_cell


@magics_class
class Chatbook(Magics):
    chatObjects = {"NONE": llm_chat('', llm_evaluator='ChatGPT')}
    dallESizes = {"small": "256x256", "medium": "512x512", "large": "1024x1024",
                  "256": "256x256", "512": "512x512", "104": "1024x1024"}

    # =====================================================
    # OpenAI
    # =====================================================
    @magic_arguments()
    @argument('-m', '--model', type=str, default="gpt-3.5-turbo-0613", help='Model')
    @argument('-t', '--temperature', type=float, default=0.7, help='Temperature (to generate responses with)')
    @argument('--top_p', default=None, help='Top probability mass')
    @argument('-n', type=int, default=1, help="Number of generated images")
    @argument('--stop', default=None, help="Number of generated images")
    @argument('--max_tokens', default=None, help='Max number of tokens')
    @argument('--response_format', type=str, default="values",
              help='Format, one of "asis", "values", or "dict"')
    @argument('-f', '--format', type=str, default='pretty',
              help="Format to display the result with; one of 'asis', 'html', 'markdown', or 'pretty'.")
    @argument('--api_key', default=None, help="API key to access the LLM service")
    @argument('--no_clipboard', action="store_true",
              help="Should the result be copied to the clipboard or not?")
    @cell_magic
    def openai(self, line, cell):
        """
        OpenAI ChatGPT magic for text generation by prompt.
        For more details about the parameters see: https://platform.openai.com/docs/api-reference/chat/create
        (Redirects to the %%chatgpt .)
        :return: LLM evaluation result.
        """
        return self.chatgpt(line, cell)

    # =====================================================
    # ChatGPT
    # =====================================================
    @magic_arguments()
    @argument('-m', '--model', type=str, default="gpt-3.5-turbo-0613", help='Model')
    @argument('-t', '--temperature', type=float, default=0.7, help='Temperature (to generate responses with)')
    @argument('--top_p', default=None, help='Top probability mass')
    @argument('-n', type=int, default=1, help="Number of generated responses.")
    @argument('--stop', default=None, help="Tokens that stop the generation when produced.")
    @argument('--max_tokens', default=None, help='Max number of tokens.')
    @argument('--response_format', type=str, default="values",
              help='LLM response format; one of "asis", "values", or "dict".')
    @argument('-f', '--format', type=str, default='pretty',
              help="Format to display the result with; one of 'asis', 'html', 'markdown', or 'pretty'.")
    @argument('--api_key', default=None, help="API key to access the LLM service.")
    @argument('--no_clipboard', action="store_true",
              help="Should the result be copied to the clipboard or not?")
    @cell_magic
    def chatgpt(self, line, cell):
        """
        OpenAI ChatGPT magic for text generation by prompt.
        For more details about the parameters see: https://platform.openai.com/docs/api-reference/chat/create
        :return: LLM evaluation result.
        """
        args = parse_argstring(self.chatgpt, line)
        args = vars(args)
        args = {k: _unquote(v) for k, v in args.items()}

        # Stop tokens
        stopTokens = args.get("stop")
        if stopTokens is None:
            stopTokens = []
        elif isinstance(stopTokens, str):
            if stopTokens.startswith("[") and stopTokens.endswith("]"):
                stopTokens = stopTokens[1:(len(stopTokens) - 1)].split(",")
                stopTokens = [_unquote(x) for x in stopTokens]
            else:
                stopTokens = [stopTokens, ]
        else:
            print(f"Cannot process the given stop tokens {stopTokens}.")

        # Max tokens
        maxTokens = args["max_tokens"]
        if isinstance(maxTokens, str):
            if maxTokens.lower() in ["none", "null"]:
                maxTokens = None
            elif isinstance(maxTokens, str):
                maxTokens = int(maxTokens)

        # Response format
        resFormat = args.get("response_format", "values")
        if resFormat not in ["asis", "values", "dict"]:
            print(
                f'The response_format argument expects a value that is one of: "asis", "values", "dict". Using "dict".')
            resFormat = "json"

        # API key
        if isinstance(args.get("api_key"), str):
            openai.api_key = args["api_key"]
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")

        if isinstance(args["top_p"], float):
            res = openai.ChatCompletion.create(
                model=args["model"],
                messages=[{"role": "user", "content": cell}],
                n=args["n"],
                top_p=args["top_p"],
                max_tokens=maxTokens,
                stop=stopTokens
            )
        else:
            res = openai.ChatCompletion.create(
                model=args["model"],
                messages=[{"role": "user", "content": cell}],
                n=args["n"],
                temperature=args["temperature"],
                max_tokens=maxTokens,
                stop=stopTokens
            )

        # Post process results
        if resFormat == "asis":
            new_cell = repr(res)
        elif resFormat in ["values", "value"]:
            new_cell = "\n".join([x["message"]["content"] for x in res.choices])
        else:
            new_cell = repr(res)

        # Copy to clipboard
        if not args.get("no_clipboard", False):
            pyperclip.copy(str(new_cell))

        # Prepare output
        # new_cell = _prep_display(new_cell, args["format"].lower())
        new_cell = _prep_result(new_cell, args["format"].lower())

        # Result
        # self.shell.run_cell(new_cell)
        return new_cell

    # =====================================================
    # DALL-E
    # =====================================================
    @magic_arguments()
    @argument('-s', '--size', type=str, default="small", help="Size of the generated image.")
    @argument('-n', type=int, default=1, help="Number of generated images.")
    @argument('--prompt', type=str, default="", help="Prompt (image edit is used if not an empty string.)")
    @argument('--mask', type=str, default="", help="File name of a mask image")
    @argument('-f', '--response_format', type=str, default="b64_json", help='Format, one of "url" or "b64_json".')
    @argument('--api_key', type=str, help="API key to access the LLM service.")
    @argument('--no_clipboard', action="store_true",
              help="Should the result be copied to the clipboard or not?")
    @cell_magic
    def dalle(self, line, cell):
        """
        OpenAI DALL-E magic for image generation by prompt.
        For more details see https://platform.openai.com/docs/api-reference/images .
        :return: Image.
        """
        args = parse_argstring(self.dalle, line)
        args = vars(args)
        args = {k: _unquote(v) for k, v in args.items()}

        # Size
        size = args.get("size", "small")
        if size not in self.dallESizes:
            expectedValues = list(self.dallESizes.keys()) + [self.dallESizes["small"], self.dallESizes["medium"],
                                                             self.dallESizes["large"]]
            print(f'The size argument expects a value that is one of: {expectedValues}. Using "256x256".')

        size = self.dallESizes.get(size, "256x256")

        # Response format
        resFormat = args.get("response_format", "b64_json")
        if resFormat not in ["url", "b64_json"]:
            print(f'The response_format argument expects a value that is one of: "url", "b64_json". Using "b64_json".')
            resFormat = "url"

        # Call to OpenAI
        if cell.strip().startswith("@") and os.path.exists(cell.strip()[1:]):
            fileName = cell.strip()[1:]

            maskFileName = args["mask"]
            if len(maskFileName.strip()) == 0:
                maskImg = None
            else:
                maskImg = open(maskFileName.strip(), "rb")

            if len(args.get("prompt", "").strip()) > 0:
                res = openai.Image.create_edit(
                    image=open(fileName, "rb"),
                    mask=maskImg,
                    prompt=args["prompt"],
                    n=args["n"],
                    size=size,
                    response_format=resFormat
                )
            else:
                res = openai.Image.create_variation(
                    image=open(fileName, "rb"),
                    n=args["n"],
                    size=size,
                    response_format=resFormat
                )
        else:
            res = openai.Image.create(
                prompt=cell,
                n=args["n"],
                size=size,
                response_format=resFormat
            )

        # Post process results
        if resFormat == "url":
            new_cell = [x["url"] for x in res["data"]]

            # Copy to clipboard
            if not args.get("no_clipboard", False):
                pyperclip.copy(new_cell)

            new_cell = f'print("{str(new_cell)}")'
        else:
            bImageData = []
            bImages = []
            for b in res["data"]:
                d = b["b64_json"]
                bImageData.append(d)
                img = f"<img src=\"data:image/png;base64,{d}\" />"
                bImages.append(img)

            # Copy to clipboard
            if not args.get("no_clipboard", False):
                if len(bImageData) == 1:
                    pyperclip.copy(bImageData[0])
                else:
                    pyperclip.copy(str(bImageData))

            new_cell = "import IPython\nIPython.display.HTML('" + ''.join(bImages) + "')"

        # Result
        self.shell.run_cell(new_cell)

    # =====================================================
    # PaLM
    # =====================================================
    @magic_arguments()
    @argument('-m', '--model', type=str, default="models/chat-bison-001", help='Model')
    @argument('-c', '--context', default=None,
              help='Text that should be provided to the model first, to ground the response.')
    @argument('-t', '--temperature', type=float, default=0.2, help='Temperature (to generate responses with).')
    @argument('--top_k', default=None, help='Sets the maximum number of tokens to sample from on each step.')
    @argument('--top_p', default=None, help='Sets the maximum cumulative probability of tokens to sample from.')
    @argument('-n', type=int, default=1, help="The maximum number of generated response messages to return.")
    @argument('--response_format', type=str, default="values",
              help='LLM response format, one of "asis", "values", or "dict".')
    @argument('-f', '--format', type=str, default='pretty',
              help="Format to display the result with; one of 'asis', 'html', 'markdown', or 'pretty'.")
    @argument('--api_key', default=None, help="API key to access the LLM service.")
    @argument('--no_clipboard', action="store_true",
              help="Should the result be copied to the clipboard or not?")
    @cell_magic
    def palm(self, line, cell):
        """
        Google's PaLM magic for image generation by prompt.
        For more details about the parameters see:
            https://developers.generativeai.google/api/python/google/generativeai/chat
        :return: LLM evaluation result.
        """
        args = parse_argstring(self.palm, line)
        args = vars(args)
        args = {k: _unquote(v) for k, v in args.items()}

        # Context
        context = args.get("context", None)
        if context is not None and not isinstance(context, str):
            print(
                f'The context argument expects a value that is a string or None. Using None.')
            context = None
        elif isinstance(context, str) and len(context.strip()) == 0:
            context = None

        # Top K
        top_k = args["top_k"]
        if isinstance(top_k, str):
            if top_k.lower() in ["none", "null"]:
                top_k = None
            elif isinstance(top_k, str):
                top_k = float(top_k)

        # Top P
        top_p = args["top_p"]
        if isinstance(top_p, str):
            if top_p.lower() in ["none", "null"]:
                top_p = None
            elif isinstance(top_p, str):
                top_p = float(top_p)

        # Response format
        resFormat = args.get("response_format", "values")
        if resFormat not in ["asis", "values", "dict"]:
            print(
                f'The response_format argument expects a value that is one of: "asis", "values", "dict". Using "dict".')
            resFormat = "json"

        # API key
        if isinstance(args.get("api_key"), str):
            google.generativeai.configure(api_key=args.get("api_key"))
        else:
            apiKey = os.environ.get("PALM_API_KEY")
            google.generativeai.configure(api_key=apiKey)

        resObj = google.generativeai.chat(
            messages=cell,
            model=args["model"],
            context=context,
            temperature=args["temperature"],
            candidate_count=args["n"],
            top_p=top_p,
            top_k=top_k
        )

        if args["n"] == 1:
            res = resObj.last
        else:
            res = [x["content"] for x in resObj.candidates]
            res = str(res)

        # Post process results
        if resFormat == "asis":
            new_cell = repr(resObj)
        elif resFormat in ["values", "value"]:
            new_cell = res
        else:
            new_cell = repr(resObj)

        # Copy to clipboard
        if not args.get("no_clipboard", False):
            pyperclip.copy(str(new_cell))

        # Prepare output
        # new_cell = _prep_display(new_cell, args["format"].lower())
        new_cell = _prep_result(new_cell, args["format"].lower())

        # Result
        # self.shell.run_cell(new_cell)
        return new_cell

    # =====================================================
    # Chat cell
    # =====================================================
    @magic_arguments()
    @argument('-i', '--chat_id', default="NONE", help="Identifier (name) of the chat object.")
    @argument('--conf', type=str, default="ChatGPT", help="LLM service access configuration.")
    @argument('--prompt', type=str, default="", help="LLM prompt.")
    @argument('--max_tokens', type=int, help="Max number of tokens for the LLM response.")
    @argument('--temperature', type=float, help="Temperature to use.")
    @argument('--api_key', type=str, help="API key to access the LLM service.")
    @argument('--echo', type=bool, default=False, help="Should the LLM evaluation steps be echoed or not?")
    @argument('-f', '--format', type=str, default='pretty',
              help="Format to display the result with; one of 'asis', 'html', 'markdown', or 'pretty'.")
    @argument('--no_clipboard', action="store_true",
              help="Should the result be copied to the clipboard or not?")
    @cell_magic
    def chat(self, line, cell):
        """
        Chat magic for interacting with LLMs.
        :return: LLM evaluation result.
        """

        args = parse_argstring(self.chat, line)
        args = vars(args)
        args = {k: _unquote(v) for k, v in args.items()}
        chatID = args.get("chat_id", "NONE")
        if chatID in self.chatObjects:
            chatObj = self.chatObjects[chatID]
        else:
            args2 = {k: v for k, v in args.items() if
                     k not in ["chat_id", "conf", "prompt", "echo", "format", "no_clipboard"]}

            # Process prompt
            prompt_spec = _unquote(args.get("prompt", ""))
            if len(prompt_spec.strip()) > 0:
                prompt_spec = llm_prompt_expand(prompt_spec, messages=[], sep="\n")

            # Make LLM configuration
            conf_spec = llm_configuration(_unquote(args["conf"]))

            # Create the chat object
            chatObj = llm_chat(prompt_spec, llm_evaluator=llm_evaluator(conf_spec, **args2))

            # Register the chat object
            self.chatObjects[chatID] = chatObj

        # Expand prompts
        res = llm_prompt_expand(cell, messages=[x["content"] for x in chatObj.messages], sep="\n")

        # Evaluate the chat message
        res = chatObj.eval(res, echo=args.get("echo", False))

        # Copy to clipboard
        if not args.get("no_clipboard", False):
            pyperclip.copy(res)

        # Prepare output
        # new_cell = _prep_display(res, args["format"].lower())
        new_cell = _prep_result(res, args["format"].lower())

        # Result
        # self.shell.run_cell(new_cell)
        return new_cell

    # =====================================================
    # Chat Meta cell
    # =====================================================
    @magic_arguments()
    @argument('-i', '--chat_id', default='NONE', type=str, help="Identifier (name) of the chat object")
    @argument('-p', '--prompt', action="store_true",
              help="Should the cell content be considered as prompt or not?")
    @argument('-c', '--conf', default='ChatGPT', type=str,
              help="Configuration to use for creating a chat object. (If --prompt is True.)")
    @argument('--all', action="store_true",
              help="Should the operation(s) be applied to all chat objects or not?")
    @cell_magic
    def chat_meta(self, line, cell):
        """
        Chat meta magic for interacting with chat objects.
        :return: LLM evaluation result.
        """

        args = parse_argstring(self.chat_meta, line)
        args = vars(args)
        chatID = args["chat_id"]
        applyToAllQ = args["all"]
        new_cell = ""
        doit = True
        cmd = cell.strip()
        if args.get("prompt", False):
            prompt_spec = cell.strip()
            prompt_spec = llm_prompt_expand(prompt_spec, messages=[], sep="\n")
            chatObj = llm_chat(prompt_spec, llm_evaluator=llm_evaluator(_unquote(args["conf"])))
            self.chatObjects[chatID] = chatObj
            new_cell = f"Created new chat object with id: ⎡{chatID}⎦\nPrompt: ⎡{prompt_spec}⎦"
        elif not applyToAllQ and chatID not in self.chatObjects:
            new_cell = f"Unknown chat object id: {chatID}."
        else:
            if cmd in ["drop", "delete"]:
                if applyToAllQ:
                    new_cell = f"Dropped all chat objects {list(self.chatObjects.keys())}."
                    self.chatObjects = {}
                else:
                    del self.chatObjects[chatID]
                    new_cell = f"Dropped the chat object {chatID}."
            elif cmd in ["clear"]:
                if applyToAllQ:
                    new_cell = f"The command 'clear' applies to chat objects only."
                else:
                    nm = len(self.chatObjects[chatID].messages)
                    self.chatObjects[chatID].messages = []
                    new_cell = f"Cleared {nm} messages of chat object {chatID}."
            elif cmd == "prompt":
                if applyToAllQ:
                    new_cell = f"The command 'prompt' applies to chat objects only."
                else:
                    new_cell = self.chatObjects[chatID].prompt()
            elif cmd in ["print", "say"]:
                if applyToAllQ:
                    new_cell = str(self.chatObjects)
                else:
                    self.chatObjects[chatID].print()
                    doit = False
            elif cmd in ["keys", "names"]:
                if applyToAllQ:
                    new_cell = str(list(self.chatObjects.keys()))
                else:
                    new_cell = str([chatID,])

        # Place result
        if doit:
            # self.shell.run_cell('print("{}".format("""' + new_cell + '"""))')
            new_cell = _prep_result(new_cell, "pretty")
            return new_cell
