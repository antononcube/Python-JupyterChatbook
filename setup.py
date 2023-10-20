import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="JupyterChatbook",
    version="0.0.16",
    author="Anton Antonov",
    author_email="antononcube@posteo.net",
    description="Custom Jupyter magics for interacting with LLMs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/antononcube/Python-JupyterChatbook",
    packages=setuptools.find_packages(),
    install_requires=["LLMFunctionObjects>=0.1.3", "LLMPrompts>=0.1.2",
                      "IPython>=8.15.0",
                      "pyperclip>=1.8.2",
                      "google-generativeai>=0.2.0",
                      "openai>=0.28.1"
                      ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
    ],
    keywords=["llm", "llm prompt", "chat object", "chatbook", "magic", "magics", "jupyter", "notebook"],
    package_data={},
    python_requires='>=3.7',
)
