"""Jupyter Chatbook magic"""
__version__ = '0.1.0'

from .Chatbook import Chatbook, initialize_chatbook


def load_ipython_extension(ipython):
    magics = ipython.register_magics(Chatbook)
    openai_images = ipython.magics_manager.magics["cell"].get("openai_images")
    if openai_images is not None:
        ipython.register_magic_function(openai_images, magic_kind="cell", magic_name="dalle")
    initialize_chatbook(ipython, chatbook=magics)
