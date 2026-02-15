from typing import Union

from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
from LLMFunctionObjects import llm_configuration, llm_evaluator, llm_chat
from LLMPrompts import llm_prompt_expand
import json
import openai
import google.generativeai
import os
import pyperclip
import IPython
import urllib.request
import urllib.error
from IPython import display
from base64 import b64decode


_INIT_ENV_VAR = "PYTHON_CHATBOOK_INIT_FILE"
_LLM_PERSONAS_ENV_VAR = "PYTHON_CHATBOOK_LLM_PERSONAS_CONF"
_DEFAULT_INIT_PATHS = ["~/.config/python-chatbook/init.py", "~/.config/init.py"]
_DEFAULT_LLM_PERSONAS_PATHS = ["~/.config/python-chatbook/llm-personas.json",
                               "~/.config/llm-personas.json"]


def _expand_path(path):
    if not isinstance(path, str):
        return None
    path = os.path.expanduser(os.path.expandvars(path.strip()))
    if not path:
        return None
    return path


def _first_existing_file(paths):
    for p in paths:
        path = _expand_path(p)
        if path and os.path.isfile(path):
            return path
    return None


def _load_init_code(ipython, path):
    if ipython is None:
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        res = ipython.run_cell(code, store_history=False)
        return bool(getattr(res, "success", False))
    except Exception as e:
        print(f"Chatbook init failed for {path}: {e}")
        return False


def _normalize_personas(data):
    personas = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str):
                spec = {"prompt": v}
            elif isinstance(v, dict):
                spec = dict(v)
            else:
                continue
            spec.setdefault("chat_id", k)
            personas.append(spec)
    elif isinstance(data, list):
        for v in data:
            if isinstance(v, dict):
                personas.append(dict(v))
    return personas


def _register_personas(personas, target=None):
    if target is None:
        target = Chatbook
    for spec in personas:
        chat_id = spec.get("chat_id") or spec.get("id") or spec.get("name")
        if not chat_id:
            continue

        prompt_spec = _unquote(spec.get("prompt", ""))
        if isinstance(prompt_spec, str) and prompt_spec.strip():
            prompt_spec = llm_prompt_expand(prompt_spec, messages=[], sep="\n")
        else:
            prompt_spec = ""

        conf_name = spec.get("conf") or spec.get("configuration") or "ChatGPT"
        conf_args = {}
        for key in ["model", "max_tokens", "temperature"]:
            if spec.get(key) is not None:
                conf_args[key] = spec[key]
        conf_spec = llm_configuration(_unquote(conf_name), **conf_args)

        evaluator_args = {}
        if isinstance(spec.get("evaluator_args"), dict):
            evaluator_args.update(spec.get("evaluator_args"))
        if spec.get("api_key") is not None:
            evaluator_args["api_key"] = spec.get("api_key")

        chat_obj = llm_chat(prompt_spec, llm_evaluator=llm_evaluator(conf_spec, **evaluator_args))
        target.chatObjects[chat_id] = chat_obj


def initialize_chatbook(ipython, chatbook=None):
    init_path = _expand_path(os.getenv(_INIT_ENV_VAR, ""))
    if init_path and os.path.isfile(init_path):
        _load_init_code(ipython, init_path)
    else:
        init_path = _first_existing_file(_DEFAULT_INIT_PATHS)
        if init_path:
            _load_init_code(ipython, init_path)

    personas_path = _expand_path(os.getenv(_LLM_PERSONAS_ENV_VAR, ""))
    if personas_path and os.path.isfile(personas_path):
        path = personas_path
    else:
        path = _first_existing_file(_DEFAULT_LLM_PERSONAS_PATHS)
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        personas = _normalize_personas(data)
        _register_personas(personas, target=chatbook if chatbook is not None else Chatbook)
    except Exception as e:
        print(f"Chatbook personas load failed for {path}: {e}")


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


def _ollama_url(host, path):
    host = host.rstrip("/")
    return f"{host}{path}"


def _ollama_request_json(url, payload=None, timeout=60, headers=None):
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
    if isinstance(headers, dict) and headers:
        for k, v in headers.items():
            if k and v:
                req.add_header(k, v)

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def _is_ollama_cloud_model(model):
    if not isinstance(model, str):
        return False
    m = model.strip().lower()
    if not m:
        return False
    return (
        m.endswith("-cloud")
        or m.endswith(":cloud")
        or m.startswith("ollama_cloud/")
        or "/cloud" in m
    )


def _is_ollama_cloud_host(host):
    if not isinstance(host, str):
        return False
    h = host.strip().lower()
    if not h:
        return False
    return "ollama.com" in h


def _ollama_get_models(host):
    try:
        data = _ollama_request_json(_ollama_url(host, "/api/tags"))
    except Exception:
        return []
    models = data.get("models", [])
    names = []
    for m in models:
        name = m.get("name") or m.get("model")
        if isinstance(name, str) and name.strip():
            names.append(name)
    return names


@magics_class
class Chatbook(Magics):
    # Lazily create chat objects to avoid importing/initializing LLM clients at import time.
    chatObjects = {}
    dallE2Sizes = {"small": "256x256", "medium": "512x512", "large": "1024x1024",
                   "256": "256x256", "512": "512x512", "104": "1024x1024", "default": "256x256", }
    dallE3Sizes = {"square": "1024x1024", "landscape": "1792x1024", "portrait": "1024x1792",
                   "1024": "1024x1024", "default": "1024x1024"}

    # =====================================================
    # OpenAI
    # =====================================================
    @magic_arguments()
    @argument('-m', '--model', type=str, default="gpt-5.2", help='Model')
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
    @argument('-m', '--model', type=str, default="gpt-4.1-mini", help='Model')
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
            res = openai.chat.completions.create(
                model=args["model"],
                messages=[{"role": "user", "content": cell}],
                n=args["n"],
                top_p=args["top_p"],
                max_tokens=maxTokens,
                stop=stopTokens
            )
        else:
            res = openai.chat.completions.create(
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
            new_cell = "\n".join([x.message.content for x in res.choices])
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
    @argument('-m', '--model', type=str, default="dall-e-2", help='Model')
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

        numberOfImages = args.get("n", 1)
        if numberOfImages < 1:
            print(f'The argument n is expected to be a positive integer, got {numberOfImages}.')
            numberOfImages = 1

        # Model
        model = args.get("model", "dall-e-2")
        if model not in ["dall-e-2", "dall-e-3"]:
            print(f'Unknown model.')

        if model == "dall-e-3" and numberOfImages > 1:
            print(f'Currently for dall-e-3 only n=1 is supported.')
            numberOfImages = 1

        # Size
        size = args.get("size", "small")
        if model == "dall-e-2":
            if size not in self.dallE2Sizes:
                expectedValues = list(self.dallE2Sizes.keys()) + [self.dallE2Sizes["small"],
                                                                  self.dallE2Sizes["medium"],
                                                                  self.dallE2Sizes["large"]]
                print(
                    f'For model dall-e-2, the size argument expects a value that is one of: {expectedValues}. Using "256x256".')

            size = self.dallE2Sizes.get(size, "256x256")
        elif model == "dall-e-3":
            if size not in self.dallE3Sizes:
                expectedValues = list(self.dallE3Sizes.keys()) + [self.dallE3Sizes["square"],
                                                                  self.dallE3Sizes["landscape"],
                                                                  self.dallE3Sizes["portrait"]]
                print(
                    f'For model dall-e-3, the size argument expects a value that is one of: {expectedValues}. Using "1024x1024".')

            size = self.dallE3Sizes.get(size, "1024x1024")

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
                res = openai.images.edit(
                    image=open(fileName, "rb"),
                    mask=maskImg,
                    prompt=args["prompt"],
                    n=numberOfImages,
                    size=size,
                    response_format=resFormat
                )
            else:
                res = openai.images.create_variation(
                    image=open(fileName, "rb"),
                    n=numberOfImages,
                    size=size,
                    response_format=resFormat
                )
        else:
            res = openai.images.generate(
                prompt=cell,
                model=model,
                n=numberOfImages,
                size=size,
                response_format=resFormat
            )

        # Post process results
        if resFormat == "url":
            new_cell = [x.url for x in res.data]

            # Copy to clipboard
            if not args.get("no_clipboard", False):
                pyperclip.copy(new_cell)

            new_cell = f'print("{str(new_cell)}")'
        else:
            bImageData = []
            bImages = []
            for b in res.data:
                d = b.b64_json
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
    # Gemini
    # =====================================================
    @magic_arguments()
    @argument('-m', '--model', type=str, default="gemini-2.5-flash", help='Model')
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
    def gemini(self, line, cell):
        """
        Google's Gemini magic for image generation by prompt.
        For more details about the parameters see:
            https://developers.generativeai.google/api/python/google/generativeai/chat
        :return: LLM evaluation result.
        """
        args = parse_argstring(self.gemini, line)
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
            apiKey = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            google.generativeai.configure(api_key=apiKey)


        model_name = args["model"]
        if model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        try:
            resObj = google.generativeai.chat(
                messages=cell,
                model=model_name,
                context=context,
                temperature=args["temperature"],
                candidate_count=args["n"],
                top_p=top_p,
                top_k=top_k
            )
        except Exception as e:
            raise e

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
    # Ollama
    # =====================================================
    @magic_arguments()
    @argument('-m', '--model', type=str, default=None,
              help='Model (if omitted, the first available model is used).')
    @argument('--host', type=str, default=None,
              help='Ollama server URL. Defaults to OLLAMA_HOST or http://localhost:11434.')
    @argument('-t', '--temperature', type=float, default=0.7, help='Temperature (to generate responses with).')
    @argument('--top_p', default=None, help='Top probability mass.')
    @argument('--top_k', default=None, help='Top-K sampling.')
    @argument('--num_predict', default=None, help='Max number of tokens.')
    @argument('--stop', default=None, help='Tokens that stop the generation when produced.')
    @argument('--response_format', type=str, default="values",
              help='LLM response format; one of "asis", "values", or "dict".')
    @argument('-f', '--format', type=str, default='pretty',
              help="Format to display the result with; one of 'asis', 'html', 'markdown', or 'pretty'.")
    @argument('--api_key', default=None, help="API key to access the Ollama cloud LLM service.")
    @argument('--no_clipboard', action="store_true",
              help="Should the result be copied to the clipboard or not?")
    @cell_magic
    def ollama(self, line, cell):
        """
        Ollama magic for text generation by prompt.
        For more details about the parameters see: https://github.com/ollama/ollama/blob/main/docs/api.md
        :return: LLM evaluation result.
        """
        args = parse_argstring(self.ollama, line)
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
            stopTokens = []

        # Max tokens
        numPredict = args["num_predict"]
        if isinstance(numPredict, str):
            if numPredict.lower() in ["none", "null"]:
                numPredict = None
            else:
                numPredict = int(numPredict)

        # Top K
        top_k = args["top_k"]
        if isinstance(top_k, str):
            if top_k.lower() in ["none", "null"]:
                top_k = None
            else:
                top_k = int(top_k)

        # Top P
        top_p = args["top_p"]
        if isinstance(top_p, str):
            if top_p.lower() in ["none", "null"]:
                top_p = None
            else:
                top_p = float(top_p)

        # Response format
        resFormat = args.get("response_format", "values")
        if resFormat not in ["asis", "values", "dict"]:
            print(
                f'The response_format argument expects a value that is one of: "asis", "values", "dict". Using "dict".')
            resFormat = "json"

        # Host and model
        host = args.get("host") or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        model = args.get("model") or os.environ.get("OLLAMA_MODEL")
        if model is None or len(str(model).strip()) == 0:
            models = [x for x in _ollama_get_models(host) if 'cloud' not in x]
            if len(models) > 0:
                model = models[0]
            else:
                model = "llama3"

        # It is somewhat of an overkill to have all these checks, but "less surprises"
        cloud_model = _is_ollama_cloud_model(model)
        cloud_host = _is_ollama_cloud_host(host)
        cloud_usage = cloud_model or cloud_host

        # API key handling
        api_key_arg = args.get("api_key")
        api_key = None
        if isinstance(api_key_arg, str) and api_key_arg.strip():
            if api_key_arg == "OLLAMA_API_KEY":
                api_key = os.environ.get("OLLAMA_API_KEY")
            else:
                api_key = api_key_arg
        if (api_key is None or len(str(api_key).strip()) == 0) and cloud_usage:
            api_key = os.environ.get("OLLAMA_API_KEY")

        options = {}
        if args.get("temperature") is not None:
            options["temperature"] = args["temperature"]
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if numPredict is not None:
            options["num_predict"] = numPredict
        if len(stopTokens) > 0:
            options["stop"] = stopTokens

        payload = {
            "model": model,
            "prompt": cell,
            "stream": False
        }
        if len(options) > 0:
            payload["options"] = options

        try:
            headers = None
            if isinstance(api_key, str) and api_key.strip() and (cloud_usage or api_key_arg is not None):
                print('HERE')
                headers = {"Authorization": f"Bearer {api_key}"}
            resObj = _ollama_request_json(
                _ollama_url(host, "/api/generate"),
                payload=payload,
                headers=headers
            )
            res = resObj.get("response", "")
        except Exception as e:
            resObj = {"error": str(e)}
            res = f"Ollama request failed: {e}"

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
        new_cell = _prep_result(new_cell, args["format"].lower())

        # Result
        return new_cell

    # =====================================================
    # Chat cell
    # =====================================================
    @magic_arguments()
    @argument('-i', '--chat_id', default="NONE", help="Identifier (name) of the chat object.")
    @argument('--conf', type=str, default="ChatGPT", help="LLM service access configuration.")
    @argument('-m', '--model', type=str, default="", help='Model')
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

            # Model
            model_spec = _unquote(args.get("model", ""))
            # Make LLM configuration
            conf_args = {}
            if len(model_spec.strip()) > 0:
                conf_args["model"] = model_spec
            if args.get("max_tokens") is not None:
                conf_args["max_tokens"] = args["max_tokens"]
            if args.get("temperature") is not None:
                conf_args["temperature"] = args["temperature"]
            conf_spec = llm_configuration(_unquote(args["conf"]), **conf_args)

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
                    new_cell = str([chatID, ])

        # Place result
        if doit:
            # self.shell.run_cell('print("{}".format("""' + new_cell + '"""))')
            new_cell = _prep_result(new_cell, "pretty")
            return new_cell
