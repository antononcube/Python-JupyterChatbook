# JupyterChatbook

Python package of a Jupyter extension that facilitates the interaction with Large Language Models (LLMs).

**Remark:** The chatbook LLM cells use the packages 
["openai"](https://pypi.org/project/openai/), [OAIp2], 
and ["google-generativeai"](https://pypi.org/project/google-generativeai/), [GAIp1].

**Remark:** The API keys for the LLM cells are taken from the Operating System (OS) environmental variables `OPENAI_API_KEY` and `PALM_API_KEY`.

**Remark:** The results of the LLM cells are automatically copied to the clipboard
using the package ["pyperclip"](https://pypi.org/project/pyperclip/), [ASp1].

--------

## Installation

### Install from GitHub

```shell
pip install -e git+https://github.com/antononcube/Python-JupyterChatbook.git#egg=Python-JupyterChatbook
```

### From PyPi

```shell
pip install JupyterChatbook
```

-------

## Demonstration notebooks (chatbooks)

| Notebook                          | Description                                                                                 |
|-----------------------------------|---------------------------------------------------------------------------------------------|
| [Chatbooks-cells-demo.ipynb](https://github.com/antononcube/Python-JupyterChatbook/blob/main/docs/Chatbook-cells-demo.ipynb) | How to do [multi-cell (notebook-wide) chats](https://www.youtube.com/watch?v=8pv0QRGc7Rw)?  |
| [Chatbook-LLM-cells.ipynb](https://github.com/antononcube/Python-JupyterChatbook/blob/main/docs/Chatbook-LLM-cells.ipynb) | How to "directly message" LLMs services?                                                    |
| [DALL-E-cells-demo.ipynb](https://github.com/antononcube/Python-JupyterChatbook/blob/main/docs/DALL-E-cells-demo.ipynb)   | How to generate images with [DALL-E](https://openai.com/dall-e-2)?                          |
| [Echoed-chats.ipynb](https://github.com/antononcube/Python-JupyterChatbook/blob/main/docs/Echoed-chats.ipynb)             | How to see the LLM interaction execution steps?                                             |


-------

## Example with chat cells

***See the notebook ["Chatbook-cells-demo.ipynb"](https://github.com/antononcube/Python-JupyterChatbook/blob/main/docs/Chatbook-cells-demo.ipynb)***

Here we start a new, named chat with specified LLM and prompt:   


```
%%chat -i yoda2 --conf ChatPaLM --prompt "You are Yoda. Respond to ALL inputs in the voice of Yoda from Star Wars. Be sure to ALWAYS use his distinctive style and syntax."
Hi! Who are you?
```

    I am Yoda, Jedi Master. I have trained many Padawans in the ways of the Force. I am old and wise, and I have seen much in my time. I am here to help you on your journey, if you will have me.


Continuing the conversation with "yoda2":


```
%%chat -i yoda2
How many students did you have?
```

    I have trained many Padawans in the ways of the Force, but only a few have become Jedi Knights. Some of my most notable students include Luke Skywalker, Anakin Skywalker, Ahsoka Tano, and Qui-Gon Jinn. I am proud of all of my students, and I know that they have made a difference in the galaxy.


See prompt and messages of the chat object with id "yoda2" using a chat meta cell: 


```
%%chat_meta yoda2
print
```

    Chat ID: 
    ------------------------------------------------------------
    Prompt:
    You are Yoda. Respond to ALL inputs in the voice of Yoda from Star Wars. Be sure to ALWAYS use his distinctive style and syntax.
    ------------------------------------------------------------
    {'role': 'user', 'content': 'Hi! Who are you?', 'timestamp': 1696015464.6843169}
    ------------------------------------------------------------
    {'role': 'assistant', 'content': 'I am Yoda, Jedi Master. I have trained many Padawans in the ways of the Force. I am old and wise, and I have seen much in my time. I am here to help you on your journey, if you will have me.', 'timestamp': 1696015466.49413}
    ------------------------------------------------------------
    {'role': 'user', 'content': 'How many students did you have?', 'timestamp': 1696015466.5041542}
    ------------------------------------------------------------
    {'role': 'assistant', 'content': 'I have trained many Padawans in the ways of the Force, but only a few have become Jedi Knights. Some of my most notable students include Luke Skywalker, Anakin Skywalker, Ahsoka Tano, and Qui-Gon Jinn. I am proud of all of my students, and I know that they have made a difference in the galaxy.', 'timestamp': 1696015474.83406}



-------

## DALL-E access

***See the notebook ["DALL-E-cells-demo.ipynb"](https://github.com/antononcube/Python-JupyterChatbook/blob/main/docs/DALL-E-cells-demo.ipynb)***

Here is a screenshot:

![](https://raw.githubusercontent.com/antononcube/Python-JupyterChatbook/main/docs/img/Python-JupyterChatbok-teaser-raccoons.png)

-------

## Implementation details

The design of this package -- and corresponding envisioned workflows with it -- follow those of
the Raku package ["Jupyter::Chatbook"](https://github.com/antononcube/Raku-Jupyter-Chatbook), [AAp3].

-------

## TODO

- [ ] TODO Implementation
  - [X] DONE PalM chat cell
  - [ ] TODO Using ["pyperclip"](https://pypi.org/project/pyperclip/)
    - [X] DONE Basic
      - [X] `%%chatgpt`
      - [X] `%%dalle`
      - [X] `%%palm`
      - [X] `%%chat`
    - [ ] TODO Switching on/off copying to the clipboard
      - [X] DONE Per cell 
        - With the argument `--copy_to_clipboard`.
      - [ ] TODO Global 
        - Can be done via the chat meta cell, but maybe a more elegant, bureaucratic solution exists.
  - [ ] TODO DALL-E image variations cell
  - [ ] TODO Mermaid-JS cell
  - [ ] TODO ProdGDT cell
  - [ ] MAYBE DeepL cell
    - See ["deepl-python"](https://github.com/DeepLcom/deepl-python)
- [ ] TODO Documentation
  - [ ] TODO Multi-cell LLM chats movie (teaser)
  - [ ] TODO Multi-cell LLM chats movie (comprehensive)
  - [ ] TODO LLM service cells movie (short)
  - [ ] TODO Code generation 

-------

## References

### Packages

[AAp1] Anton Antonov,
[LLMFunctionObjects Python package](https://github.com/antononcube/Python-packages/tree/main/LLMFunctionObjects),
(2023),
[Python-packages at GitHub/antononcube](https://github.com/antononcube/Python-packages).

[AAp2] Anton Antonov,
[LLMPrompts Python package](https://github.com/antononcube/Python-packages/tree/main/LLMPrompts),
(2023),
[Python-packages at GitHub/antononcube](https://github.com/antononcube/Python-packages).

[AAp3] Anton Antonov,
[Jupyter::Chatbook Raku package](https://github.com/antononcube/Raku-Jupyter-Chatbook,
(2023),
[GitHub/antononcube](https://github.com/antononcube).

[ASp1] Al Sweigart, 
[pyperclip (Python package)](https://pypi.org/project/pyperclip/),
(2013-2021),
[PyPI.org/AlSweigart](https://pypi.org/user/AlSweigart/).

[GAIp1] Google AI,
[google-generativeai (Google Generative AI Python Client)](https://pypi.org/project/google-generativeai/),
(2023),
[PyPI.org/google-ai](https://pypi.org/user/google-ai/).

[OAIp1] OpenAI, 
[openai (OpenAI Python Library)](https://pypi.org/project/openai/),
(2020-2023),
[PyPI.org](https://pypi.org/).

### Videos

[AAv1] Anton Antonov,
["Jupyter Chatbook multi cell LLM chats teaser (Raku)"](https://www.youtube.com/watch?v=wNpIGUAwZB8),
(2023),
[YouTube/@AAA4Prediction](https://www.youtube.com/@AAA4prediction).

[AAv2] Anton Antonov,
["Jupyter Chatbook multi cell LLM chats teaser (Python)"](https://www.youtube.com/watch?v=8pv0QRGc7Rw),
(2023),
[YouTube/@AAA4Prediction](https://www.youtube.com/@AAA4prediction).



