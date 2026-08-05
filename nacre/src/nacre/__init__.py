import jax

jax.config.update("jax_enable_x64", True)
assert jax.config.x64_enabled, "nacre requires JAX x64 mode"
