import chex
import jax.numpy as jnp
from jax import grad, jit
import jax.tree_util as jtu
import optax
from typing import NamedTuple
import logstate
from optimizer.online_learners import OnlineLearner

class ADAMState(NamedTuple):
    count: chex.Array
    m_t: chex.Array  # First moment vector
    v_t: chex.Array  # Second moment vector (uncentered variance)
    logging: logstate.Log  # Logging object to track metrics
 
def adam2(
    lr: optax.ScalarOrSchedule,
    beta1: float = 0.9,  # Momentum for the first moment
    beta2: float = 0.999,  # Momentum for the second moment
    epsilon: float = 1e-8  # Numerical stability in division
) -> OnlineLearner:
    
    class LogChecker(object):
        """A dummy class to make sure all logs have the same structure and data type."""
        def __init__(self):
            self.logging = {
                
            }

        def __call__(self, **kwargs):
            for key, value in kwargs.items():
                if key in self.logging:
                    self.logging[key] = value
                else:
                    raise ValueError(f"Alert: Item '{key}' either does not exist in logging or is not compatible.")
            return logstate.Log(self.logging)

    logger = LogChecker()

    def init_fn(params):
        initial_m = jtu.tree_map(jnp.zeros_like, params)
        initial_v = jtu.tree_map(jnp.zeros_like, params)
        logger = LogChecker()
        return ADAMState(count=jnp.zeros([], jnp.int32), m_t=initial_m, v_t=initial_v, logging=logger())

    def update_fn(updates, state, params):
        count_inc = optax.safe_int32_increment(state.count)

        m_t_new = jtu.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, state.m_t, updates)
        v_t_new = jtu.tree_map(lambda v, g: beta2 * v + (1 - beta2) * g**2, state.v_t, updates)

        # Bias-corrected first and second moments
        m_hat = jtu.tree_map(lambda m: m / (1 - beta1 ** count_inc), m_t_new)
        v_hat = jtu.tree_map(lambda v: v / (1 - beta2 ** count_inc), v_t_new)

        eta_t = lr if callable(lr) else lr
        scaled_learning_rate = jtu.tree_map(lambda v: eta_t / (jnp.sqrt(v) + epsilon), v_hat)
        parameter_updates = jtu.tree_map(lambda m, s_lr: -s_lr * m, m_hat, scaled_learning_rate)

        params_next = jtu.tree_map(lambda p, u: p + u, params, parameter_updates)

        return params_next, ADAMState(
            count=count_inc,
            m_t=m_t_new,
            v_t=v_t_new,
            logging=logger(**{
                })
        )

    return OnlineLearner(init_fn, update_fn)
