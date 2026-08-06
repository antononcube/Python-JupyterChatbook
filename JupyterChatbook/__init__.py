"""Jupyter Chatbook magic"""
__version__ = '0.1.0'

from .Chatbook import Chatbook, initialize_chatbook


def load_ipython_extension(ipython):
    magics = ipython.register_magics(Chatbook)
    xai_images = ipython.magics_manager.magics["cell"].get("xai_images")
    if xai_images is not None:
        ipython.register_magic_function(xai_images, magic_kind="cell", magic_name="xai-images")
    initialize_chatbook(ipython, chatbook=magics)
