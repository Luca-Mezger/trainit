from jax import numpy as jnp
import chex
import optax
from typing import NamedTuple
from optimizer.online_learners import OnlineLearner
from jax import tree_util as jtu

class EGState(NamedTuple):
    count: chex.Array  # stores count as an array
    weights: chex.Array  # stores the weights for the exponentiated gradient update

def exponentiated_gradient(learning_rate: optax.ScalarOrSchedule) -> OnlineLearner:
    """Exponentiated Gradient algorithm."""

    def init_fn(params):
        # Ensure params are initialized and are JAX arrays
        if params is None:
            raise ValueError("Initial weights must be provided")
        return EGState(count=jnp.zeros([], jnp.int32), weights=params)

    def update_fn(updates, state, params):
        eta = learning_rate(state.count) if callable(learning_rate) else learning_rate
        
        # Apply the exponentiated gradient update using tree_map for compatibility
        weights_exp = jtu.tree_map(lambda w, u: w * jnp.exp(-eta * u), state.weights, updates)
        
        # Use tree_map to safely compute the sum across all dimensions
        sum_weights = jtu.tree_map(jnp.sum, weights_exp)  # This might need adjustment based on your specific structure

        # Normalize weights across all dimensions using tree utilities
        new_weights = jtu.tree_map(lambda w, s: w / s, weights_exp, sum_weights)

        # Increment the count
        new_count = optax.safe_int32_increment(state.count)
            
        return new_weights, EGState(count=new_count, weights=new_weights)
    
    return OnlineLearner(init_fn, update_fn)
