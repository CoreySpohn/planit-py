"""Test configuration: plan tables hold float64 start epochs."""

import jax

jax.config.update("jax_enable_x64", True)
