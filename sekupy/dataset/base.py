"""`Dataset` lives in `sekupy.dataset.dataset`; this module re-exports it
under its long-established public import path (`sekupy.dataset.base.Dataset`,
used throughout the example gallery and docstrings).
"""
from sekupy.dataset.dataset import AttrDict, Dataset

__all__ = ["Dataset", "AttrDict"]
