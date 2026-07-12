"""Compatibility shim for running SeismicFoundationModel (SFM) on modern PyTorch.

Auto-loaded by Python at interpreter startup when this directory is on PYTHONPATH.
It recreates the ``torch._six`` module that SFM's ``util/misc.py`` imports
(``from torch._six import inf``). ``torch._six`` was removed in PyTorch 2.0, but
the repo pins PyTorch 1.8.1. Because this GPU is Blackwell (sm_120) it *requires*
PyTorch >= 2.7, so we provide the missing module here instead of editing the repo.

No files inside tools/seismicfoundationmodel/ are modified.
"""
import sys
import math
import types


def _install_torch_six():
    try:
        import torch._six  # noqa: F401
        return  # already present (old torch) -> nothing to do
    except ModuleNotFoundError:
        pass
    import collections.abc as _abc

    m = types.ModuleType("torch._six")
    m.inf = math.inf
    m.nan = math.nan
    m.container_abcs = _abc
    m.string_classes = (str, bytes)
    m.int_classes = int
    sys.modules["torch._six"] = m


_install_torch_six()
