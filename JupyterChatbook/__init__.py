"""Jupyter Chatbook magic"""
__version__ = '0.1.0'

from .Chatbook import Chatbook, initialize_chatbook


def load_ipython_extension(ipython):
    magics = ipython.register_magics(Chatbook)
    initialize_chatbook(ipython, chatbook=magics)
