# JupyterChatbook

Python package of a Jupyter extension that facilitates the interaction with Large Language Models (LLMs).


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

Here is screenshot:

![](https://raw.githubusercontent.com/antononcube/Python-JupyterChatbook/main/docs/img/Python-JupyterChatbok-teaser-raccoons.png)

-------

## References

### Packages

[AAp1] Anton Antonov,
[LLMFunctionObjects Python package](https://github.com/antononcube/Python-packages/tree/main/LLMFunctionObjects),
(2023),
[Python-packages at GitHub/antononcube](https://github.com/antononcube/Python-packages).

[AAp2] Anton Antonov,
[LLMPrompts Python package](hhttps://github.com/antononcube/Python-packages/tree/main/LLMPrompts),
(2023),
[Python-packages at GitHub/antononcube](https://github.com/antononcube/Python-packages).

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
["Jupyter Chatbook multi cell LLM chats teaser (Python)"](),
(2023),
[YouTube/@AAA4Prediction](https://www.youtube.com/@AAA4prediction).
***TBD...***



