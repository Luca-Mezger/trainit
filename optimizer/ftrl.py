from jax import numpy as jnp
from jax import tree_util as jtu  # imports tree utilities from jax for operations on pytrees
import chex  # imports chex for JAX type and shape checking
import jax
import optax  # imports optax for optimization tools
from typing import NamedTuple  # imports NamedTuple for creating custom tuple types
from optimizer.online_learners import OnlineLearner
from jax import grad, jit
import logstate
import utils


class FTRL3State(NamedTuple):
    count: chex.Array
    weighted_sum: chex.Array
    sum_squared: chex.Array
    logging: logstate.Log  # Logging object to track metrics

def ftrl(
    lr: optax.ScalarOrSchedule,
    beta_1: float = 0.99,
    beta_2: float = 0.9801, #beta_2 = beta_1^2
    epsilon: float = 1e-8,
) -> OnlineLearner:

    class LogChecker(object):
        """A dummy class to make sure all logs have the same structure and data type."""
        def __init__(self):
            self.logging = {}

        def __call__(self, **kwargs):
            for key, value in kwargs.items():
                if key in self.logging:
                    self.logging[key] = value
                else:
                    raise ValueError(f"Alert: Item '{key}' either does not exist in logging or is not compatible.")
            return logstate.Log(self.logging)

    logger = LogChecker()

    def init_fn(params):
        initial_weighted_sum = jtu.tree_map(jnp.zeros_like, params)
        initial_sum_squared = jtu.tree_map(jnp.zeros_like, params)
        return FTRL3State(
            count=jnp.zeros([], jnp.int32),
            weighted_sum=initial_weighted_sum,
            sum_squared=initial_sum_squared,
            logging=logger()
        )

    def update_fn(updates, state, params):
        if callable(lr):
            eta = lr(state.count)
        else:
            eta = lr
        count_inc = optax.safe_int32_increment(state.count)

        new_weighted_sum = jtu.tree_map(lambda ws, g: beta_1 * ws + g, state.weighted_sum, updates)
        new_sum_squared = jtu.tree_map(lambda ss, g: beta_2 * ss + (g ** 2), state.sum_squared, updates)

        scale = jtu.tree_map(lambda ws, ss: eta * ws / jnp.sqrt(ss + epsilon), new_weighted_sum, new_sum_squared)

        params_next = jtu.tree_map(lambda s: -s, scale)
        
        return params_next, FTRL3State(
            count=count_inc,
            weighted_sum=new_weighted_sum,
            sum_squared=new_sum_squared,
            logging=logger()
        )

    return OnlineLearner(init_fn, update_fn)



class DB_FTRLState(NamedTuple):
    count: chex.Array
    weighted_sum: chex.Array
    sum_squared: chex.Array
    logging: logstate.Log  # Logging object to track metrics

def debiase_ftrl( #with additional debiasing term like in ADAM
    lr: optax.ScalarOrSchedule,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
) -> OnlineLearner:

    class LogChecker(object):
        """A dummy class to make sure all logs have the same structure and data type."""
        def __init__(self):
            self.logging = {}

        def __call__(self, **kwargs):
            for key, value in kwargs.items():
                if key in self.logging:
                    self.logging[key] = value
                else:
                    raise ValueError(f"Alert: Item '{key}' either does not exist in logging or is not compatible.")
            return logstate.Log(self.logging)

    logger = LogChecker()

    def init_fn(params):
        initial_weighted_sum = jtu.tree_map(jnp.zeros_like, params)
        initial_sum_squared = jtu.tree_map(jnp.zeros_like, params)
        return DB_FTRLState(
            count=jnp.zeros([], jnp.int32),
            weighted_sum=initial_weighted_sum,
            sum_squared=initial_sum_squared,
            logging=logger()
        )

    def update_fn(updates, state, params):
        if callable(lr):
            eta = lr(state.count)
        else:
            eta = lr
        count_inc = optax.safe_int32_increment(state.count)

        new_weighted_sum = jtu.tree_map(lambda ws, g: beta_1 * ws + g, state.weighted_sum, updates)
        new_sum_squared = jtu.tree_map(lambda ss, g: beta_2 * ss + (g ** 2), state.sum_squared, updates)

        #debias_weighted_sum = jtu.tree_map(lambda ws: ws / (1 - beta_1 ** count_inc), new_weighted_sum)
        #debias_sum_squared = jtu.tree_map(lambda ss: ss / (1 - beta_2 ** count_inc), new_sum_squared)
        debias_weighted_sum = jtu.tree_map(lambda ws: ws, new_weighted_sum)
        debias_sum_squared = jtu.tree_map(lambda ss: ss, new_sum_squared)

        scale = jtu.tree_map(lambda ws, ss: eta * ws / jnp.sqrt(ss + epsilon), debias_weighted_sum, debias_sum_squared)

        params_next = jtu.tree_map(lambda s: -s, scale)
        
        return params_next, DB_FTRLState(
            count=count_inc,
            weighted_sum=new_weighted_sum,
            sum_squared=new_sum_squared,
            logging=logger()
        )

    return OnlineLearner(init_fn, update_fn)




class FTRL_WState(NamedTuple):
    count: chex.Array
    weighted_sum: chex.Array
    sum_squared: chex.Array
    logging: logstate.Log  # Logging object to track metrics

def ftrlW( #with weight decay
    lr: optax.ScalarOrSchedule,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    weight_decay_lambda: float = 0.001,
    epsilon: float = 1e-8,
) -> OnlineLearner:

    class LogChecker(object):
        """A dummy class to make sure all logs have the same structure and data type."""
        def __init__(self):
            self.logging = {}

        def __call__(self, **kwargs):
            for key, value in kwargs.items():
                if key in self.logging:
                    self.logging[key] = value
                else:
                    raise ValueError(f"Alert: Item '{key}' either does not exist in logging or is not compatible.")
            return logstate.Log(self.logging)

    logger = LogChecker()

    def init_fn(params):
        initial_weighted_sum = jtu.tree_map(jnp.zeros_like, params)
        initial_sum_squared = jtu.tree_map(jnp.zeros_like, params)
        return FTRL3State(
            count=jnp.zeros([], jnp.int32),
            weighted_sum=initial_weighted_sum,
            sum_squared=initial_sum_squared,
            logging=logger()
        )

    def update_fn(updates, state, params):
        if callable(lr):
            eta = lr(state.count)
        else:
            eta = lr
        count_inc = optax.safe_int32_increment(state.count)

        new_weighted_sum = jtu.tree_map(lambda ws, g: beta_1 * ws + g, state.weighted_sum, updates)
        new_sum_squared = jtu.tree_map(lambda ss, g: beta_2 * ss + (g ** 2), state.sum_squared, updates)

        scale = jtu.tree_map(lambda ws, ss: eta * ws / jnp.sqrt(ss + epsilon), new_weighted_sum, new_sum_squared)

        #weight decay
        params_next = jtu.tree_map(lambda s, p: -s - eta * weight_decay_lambda * p, scale, params)
        
        return params_next, FTRL3State(
            count=count_inc,
            weighted_sum=new_weighted_sum,
            sum_squared=new_sum_squared,
            logging=logger()
        )

    return OnlineLearner(init_fn, update_fn)

