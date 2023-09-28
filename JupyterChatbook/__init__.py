"""Jupyter Chatbook magic"""
__version__ = '0.1.0'

from .Chatbook import Chatbook


def load_ipython_extension(ipython):
    ipython.register_magics(Chatbook)
