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

class OMDState(NamedTuple):
    count: chex.Array 
def omd(
    learning_rate: optax.ScalarOrSchedule,
    beta: float = 0.9
) -> OnlineLearner:
    """Online Mirror Descent (OMD) with Euclidean norm as the mirror map."""

    @jit
    def euclidean_mirror_map(x):
        return 0.5 * sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(x))

    grad_mirror_map = jit(grad(euclidean_mirror_map))
    # for Euclidean norm: inverse = idendity
    inverse_grad_mirror_map = jit(lambda y: y)

    def init_fn(params=None):
        return OMDState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params):
        if callable(learning_rate):
            lr = learning_rate(state.count)
        else:
            lr = learning_rate

        current_grad = grad_mirror_map(params)
        gradient_updates = jtu.tree_map(lambda u: lr * u, updates)
        dual_variable = jtu.tree_map(lambda cg, gu: beta*(cg - gu), current_grad, gradient_updates) #TODO: beta value in config

        params_next = jtu.tree_map(inverse_grad_mirror_map, dual_variable)
        #no projection
        count_inc = optax.safe_int32_increment(state.count)
        return params_next, OMDState(count=count_inc)

    return OnlineLearner(init_fn, update_fn)


class OOMDState(NamedTuple):
    count: chex.Array
    hint: chex.Array  # Store the previous hint
    logging: logstate.Log  # Logging object to track metrics

def oomd(
    learning_rate: optax.ScalarOrSchedule,
    beta: float = 0.9  # Momentum term for the previous hint
) -> OnlineLearner:
    
    class LogChecker(object):
        """A dummy class to make sure all logs have the same structure and data type."""
        def __init__(self):
            self.logging = {
                "hint/norm(prev_hint, grad)": jnp.zeros([]),
                "hint/<prev_hint, grad>": jnp.zeros([]),
            }

        def __call__(self, **kwargs):
            for key, value in kwargs.items():
                if key in self.logging:
                    self.logging[key] = value
                else:
                    raise ValueError(f"Alert: Item '{key}' either does not exist in logging or is not compatible.")
            return logstate.Log(self.logging)

    logger = LogChecker()

    @jit
    def euclidean_mirror_map(params):
        """Euclidean mirror map Φ(x) = 0.5 * ||x||^2."""
        #return 0.5 * sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(params))
        return 0.5 * jtu.tree_reduce(jnp.add, jtu.tree_map(lambda x: jnp.sum(jnp.square(x)), params))

    grad_mirror_map = jit(grad(euclidean_mirror_map))
    inverse_grad_mirror_map = jit(lambda y: y)

    def init_fn(params=None):
        initial_hint = jtu.tree_map(lambda x: jnp.zeros_like(x), params) if params is not None else None
        return OOMDState(
            count=jnp.zeros([], jnp.int32),
            hint=initial_hint,
            logging=logger(),
        )

    def update_fn(updates, state, params):
        if callable(learning_rate):
            lr = learning_rate(state.count)
        else:
            lr = learning_rate

        current_grad = grad_mirror_map(params)
        previous_hint = state.hint
        next_hint = jtu.tree_map(lambda ph, h: beta * ph + (1 - beta) * h, previous_hint, updates) #h_{t+1} = β * h_t + (1 - β) * g_t
        #next_hint = updates
        # Update rule: x_{t+1} = x_t - learning_rate * (g_t + (h_t - h_{t+1}))
        combined_updates = jtu.tree_map(lambda u, nh, ph: lr * (u + (ph - nh)), updates, next_hint, previous_hint)
        dual_variable = jtu.tree_map(lambda cg, cu: 0.99*(cg - cu), current_grad, combined_updates)
        params_next = jtu.tree_map(inverse_grad_mirror_map, dual_variable)
        
        count_inc = optax.safe_int32_increment(state.count)
        
        #logging
        hint_distance = utils.tree_l2_norm(jax.tree_map(lambda x, y: x - y, previous_hint, current_grad))
        hint_inner_product = utils.tree_inner_product(previous_hint, updates)


        return params_next, OOMDState(
            count=count_inc,
            hint=next_hint,
            logging=logger(**{
                "hint/norm(prev_hint, grad)": hint_distance,
                "hint/<prev_hint, grad>": hint_inner_product
            })
        )

    return OnlineLearner(init_fn, update_fn)


class AOOMDState(NamedTuple):
    count: chex.Array
    hint: chex.Array  # Store the previous hint
    v_t: chex.Array
    logging: logstate.Log  # Logging object to track metrics

def adaptive_oomd(
    D: optax.ScalarOrSchedule,
    beta: float = 0.9,  # Momentue previous hint term for th
    beta_2: float = 0.9, #beta for lr calculation
) -> OnlineLearner:
    
    class LogChecker(object):
        """A dummy class to make sure all logs have the same structure and data type."""
        def __init__(self):
            self.logging = {
                "hint/norm(prev_hint, grad)": jnp.zeros([]),
                "hint/<prev_hint, grad>": jnp.zeros([]),
                "learning_rate/norm(eta_t)": jnp.zeros([]),
                "learning_rate/avg(eta_t)": jnp.zeros([]),
                "updates/avg(v_t_new)": jnp.zeros([]),
            }

        def __call__(self, **kwargs):
            for key, value in kwargs.items():
                if key in self.logging:
                    self.logging[key] = value
                else:
                    raise ValueError(f"Alert: Item '{key}' either does not exist in logging or is not compatible.")
            return logstate.Log(self.logging)

    logger = LogChecker()

    @jit
    def euclidean_mirror_map(params):
        """Euclidean mirror map Φ(x) = 0.5 * ||x||^2."""
        #return 0.5 * sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(params))
        return 0.5 * jtu.tree_reduce(jnp.add, jtu.tree_map(lambda x: jnp.sum(jnp.square(x)), params))

    grad_mirror_map = jit(grad(euclidean_mirror_map))
    inverse_grad_mirror_map = jit(lambda y: y)

    def init_fn(params=None):
        initial_hint = jtu.tree_map(lambda x: jnp.zeros_like(x), params) if params is not None else None
        initial_v_t = jtu.tree_map(lambda x: jnp.zeros_like(x), params) if params is not None else None
        return AOOMDState(
            count=jnp.zeros([], jnp.int32),
            hint=initial_hint,
            v_t=initial_v_t,
            logging=logger(),
        )

    def update_fn(updates, state, params):
        if callable(D):
            eta = D(state.count)
        else:
            eta = D
    
        #scalar learning rate
        g_t = utils.tree_l2_norm(updates)
        v_t_new = beta_2 * state.v_t + (1 - beta_2) * g_t ** 2
        eta_t = D / (2 * jnp.sqrt(v_t_new)) #dont name it D, instead eta and name eta_t different

        current_grad = grad_mirror_map(params)
        previous_hint = state.hint
        next_hint = jtu.tree_map(lambda ph, h: beta * ph + (1 - beta) * h, previous_hint, updates) #h_{t+1} = β * h_t + (1 - β) * g_t
        # Update rule: x_{t+1} = x_t - learning_rate * (g_t + (h_t - h_{t+1}))
        combined_updates = jtu.tree_map(lambda u, nh, ph: eta_t * (u + (ph - nh)), updates, next_hint, previous_hint)
        dual_variable = jtu.tree_map(lambda cg, cu: (cg - cu), current_grad, combined_updates)
        params_next = jtu.tree_map(inverse_grad_mirror_map, dual_variable)
        
        """#coordinate wise
        g_t_squared = jtu.tree_map(lambda x: x ** 2, updates)
        v_scaled = jtu.tree_map(lambda v: utils.tree_scalar_multiply(v, beta_2), state.v_t)
        g_scaled = jtu.tree_map(lambda g: utils.tree_scalar_multiply(g, (1 - beta_2)), g_t_squared)
        v_t_new = jtu.tree_map(lambda v, g: v + g, v_scaled, g_scaled)
        #v_t_new = jtu.tree_map(lambda v: v / (1 - beta_2 ** state.count), v_t_new) #bias correction

        eta_t = jtu.tree_map(lambda v: eta / (2 * jnp.sqrt(1e-8 + v)), v_t_new)
 
        current_grad = grad_mirror_map(params)
        previous_hint = state.hint
        next_hint = jtu.tree_map(lambda ph, h: beta * ph + (1 - beta) * h, previous_hint, updates) #h_{t+1} = β * h_t + (1 - β) * g_t
        #next_hint = jtu.tree_map(lambda h: h / (1 - beta ** state.count), next_hint) #bias correction

        #Update rule: x_{t+1} = x_t - learning_rate * (g_t + (h_t - h_{t+1}))
        hint_differences = jtu.tree_map(lambda nh, ph: nh - ph, next_hint, previous_hint)
        updated_with_hints = jtu.tree_map(lambda u, hd: u + hd, updates, hint_differences)
        combined_updates = jtu.tree_map(lambda eta, uwh: eta * uwh, eta_t, updated_with_hints)
        dual_variable = jtu.tree_map(lambda cg, cu: (cg - cu), current_grad, combined_updates)
        params_next = jtu.tree_map(inverse_grad_mirror_map, dual_variable)
         """
        count_inc = optax.safe_int32_increment(state.count)
         
        #logging
        hint_distance = utils.tree_l2_norm(jax.tree_map(lambda x, y: x - y, previous_hint, updates))
        hint_inner_product = utils.tree_inner_product(previous_hint, updates)


        return params_next, AOOMDState(
            count=count_inc,
            hint=next_hint,
            v_t=v_t_new,
            logging=logger(**{
                "hint/norm(prev_hint, grad)": hint_distance,
                "hint/<prev_hint, grad>": hint_inner_product,
                "learning_rate/norm(eta_t)": utils.tree_l2_norm(eta_t), #norm
                "learning_rate/avg(eta_t)": jnp.mean(jnp.concatenate(jtu.tree_leaves(jtu.tree_map(jnp.ravel, eta_t)))),
                "updates/avg(v_t_new)": jnp.mean(jnp.concatenate(jtu.tree_leaves(jtu.tree_map(jnp.ravel, v_t_new))))
            })
        )

    return OnlineLearner(init_fn, update_fn)
