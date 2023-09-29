import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="JupyterChatbook",
    version="0.0.1",
    author="Anton Antonov",
    author_email="antononcube@posteo.net",
    description="Custom Jupyter magics for interacting with LLMs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/antononcube/Python-JupyterChatbook",
    packages=setuptools.find_packages(),
    install_requires=["LLMFunctionObjects", "LLMPrompts", "IPython"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
    ],
    keywords=["llm", "chat object", "chatbook", "magic", "magics", "jupyter", "notebook"],
    package_data={},
    python_requires='>=3.7',
)
