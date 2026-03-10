# JupyterChatbook Cheatsheet

Quick reference for the Python Jupyter extension in this repo.

## 0) Load extension in notebook

```python
%load_ext JupyterChatbook
```

This registers the magics and runs automatic initialization (`init.py` + personas JSON if configured/found).

## 1) New LLM persona initialization

### A) Create persona with `%%chat` (and immediately send first message)

```python
%%chat -i assistant1 --conf ChatGPT --model gpt-4.1-mini --prompt "You are a concise technical assistant."
Say hi and ask what I am working on.
```

### B) Create persona with `%%chat_meta --prompt` (create only)

```python
%%chat_meta -i assistant2 --prompt --conf ChatGPT --model gpt-4.1-mini
You are a code reviewer focused on correctness and edge cases.
```

You can use prompt specs from `LLMPrompts`, for example:

```python
%%chat_meta -i yoda --prompt
@Yoda
```

## 2) Notebook-wide chat with an LLM persona

### Continue an existing chat object

```python
%%chat -i assistant1
Give me a 5-step implementation plan for adding authentication to a FastAPI app.
```

```python
%%chat -i assistant1
Now rewrite step 2 with test-first details.
```

### Default chat object (`NONE`)

```python
%%chat
Summarize the previous answer in 3 bullets.
```

## 3) Management of personas (`%%chat_meta`)

### Query one persona

```python
%%chat_meta -i assistant1
prompt
```

```python
%%chat_meta -i assistant1
print
```

### Query all personas

```python
%%chat_meta --all
keys
```

```python
%%chat_meta --all
print
```

### Delete one persona

```python
%%chat_meta -i assistant1
delete
```

### Delete all personas

```python
%%chat_meta --all
delete
```

### Clear message history of one persona (keep persona)

```python
%%chat_meta -i assistant2
clear
```

`%%chat_meta` command aliases:
- `delete` or `drop`
- `keys` or `names`
- `print` or `say`

## 4) Regular chat cells vs direct LLM-provider cells

### Regular chat cells (`%%chat`)

- Stateful across cells (conversation memory stored in chat objects).
- Persona-oriented via `--chat_id` + optional `--prompt`.
- Backend chosen with `--conf` (default: `ChatGPT`).

### Direct provider cells (`%%chatgpt`, `%%openai`, `%%gemini`, `%%ollama`, `%%dalle`)

- Direct single-call access to provider APIs.
- Useful for explicit provider/model control.
- Do not use chat-object memory managed by `%%chat`.

Examples:

```python
%%chatgpt --model gpt-4.1-mini
Write a regex for US ZIP+4.
```

```python
%%gemini --model gemini-2.5-flash
Explain async/await in Python in plain language.
```

```python
%%ollama --model llama3
Give me three Linux troubleshooting tips.
```

```python
%%dalle --model dall-e-3 --size square
A watercolor painting of a lighthouse in stormy weather.
```

## 5) LLM provider access facilitation

API keys can be passed inline (`--api_key`) or through environment variables.

### Notebook-session environment setup

```python
%env OPENAI_API_KEY=YOUR_OPENAI_KEY
%env GEMINI_API_KEY=YOUR_GEMINI_KEY
%env OLLAMA_API_KEY=YOUR_OLLAMA_KEY
```

or:

```python
import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_KEY"
os.environ["OLLAMA_API_KEY"] = "YOUR_OLLAMA_KEY"
```

Ollama-specific defaults:

- `OLLAMA_HOST` (default host fallback is `http://localhost:11434`)
- `OLLAMA_MODEL` (default model if `--model` not given)

## 6) Notebook/chatbook session initialization with custom code + personas JSON

Initialization runs when the extension is loaded.

### A) Custom Python init code

- Env var override: `PYTHON_CHATBOOK_INIT_FILE`
- If not set, first existing file is used in this order:
1. `~/.config/python-chatbook/init.py`
2. `~/.config/init.py`

Use this for imports/helpers you always want in chatbook sessions.

### B) Pre-load personas from JSON

- Env var override: `PYTHON_CHATBOOK_LLM_PERSONAS_CONF`
- If not set, first existing file is used in this order:
1. `~/.config/python-chatbook/llm-personas.json`
2. `~/.config/llm-personas.json`

Supported JSON shapes:

### Shape 1: object (keys become `chat_id`)

```json
{
  "writer": {
    "conf": "ChatGPT",
    "prompt": "@CodeWriterX|Python",
    "model": "gpt-4.1-mini",
    "max_tokens": 4096,
    "temperature": 0.4
  },
  "editor": "You are a strict copy editor."
}
```

### Shape 2: list of persona specs

```json
[
  {
    "chat_id": "python",
    "conf": "ChatGPT",
    "prompt": "@CodeWriterX|Python",
    "model": "gpt-4.1-mini",
    "max_tokens": 8192,
    "temperature": 0.4
  }
]
```

Recognized persona spec fields include:
- `chat_id` (or `id`, `name`)
- `prompt`
- `conf` (or `configuration`)
- `model`, `max_tokens`, `temperature`, `base_url`
- `api_key`
- `evaluator_args` (object)

Verify pre-loaded personas:

```python
%%chat_meta --all
keys
```
