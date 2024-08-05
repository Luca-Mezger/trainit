from jax import numpy as jnp  
from jax import tree_util as jtu  # imports tree utilities from jax for operations on pytrees
import chex  # imports chex for JAX type and shape checking
import optax  # imports optax for optimization tools
from typing import NamedTuple  # imports NamedTuple for creating custom tuple types
from optimizer.online_learners import OnlineLearner

class ZEROState(NamedTuple):  # defines a new named tuple for storing optimizer state
    """ZERO state."""
    count: chex.Array  # stores count as an array

def zero(
    learning_rate: optax.ScalarOrSchedule,  # parameter, not used in this function
    weight_decay: float = 0.0,  # parameter, not used in this function
) -> OnlineLearner:  # function returns an OnlineLearner object
    """Online Gradient Descent (zero) simplified to zero updates.

    Args:
        learning_rate: ignored here
        weight_decay: ignored here
    """

    def init_fn(params=None):  # initializes the optimizer state
        del params  # deletes params to indicate they are not used
        return ZEROState(count=jnp.zeros([], jnp.int32))  # returns ZEROState with count set to zero
    
    def update_fn(updates, state, params):  # defines the update function
        zero_updates = jtu.tree_map(lambda x: jnp.zeros_like(x), params)  # sets all updates to zero
        count_inc = optax.safe_int32_increment(state.count)  # safely increments the count
        return zero_updates, ZEROState(count=count_inc)  # returns zero updates and the new state
    
    return OnlineLearner(init_fn, update_fn)  # returns the OnlineLearner object with init and update functions
