"""Test configuration: plan tables hold float64 start epochs."""

import importlib.util

import jax

jax.config.update("jax_enable_x64", True)

pytest_plugins = []
if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")
