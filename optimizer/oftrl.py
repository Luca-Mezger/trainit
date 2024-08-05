from jax import numpy as jnp
from jax import tree_util as jtu
from jax import jax
import chex
import optax
from typing import NamedTuple
from optimizer.online_learners import OnlineLearner, OnlineLearnerExtraArgs
import logstate
import utils

def update_hint(method, previous_hint, updates, count_inc, prev_params, m_t, beta_3=0.9):
    if method == 0:
        # 0. easy test
        hint = updates
    elif method == 1:
        # 1. exponential moving Average, ema
        hint = jtu.tree_map(lambda ph, h: beta_3 * ph + (1 - beta_3) * h, previous_hint, updates)
    elif method == 2:
        # 2. momentum
        hint = jtu.tree_map(lambda ph, h: ph + (1 - beta_3) * (h - ph), previous_hint, updates)
    elif method == 3:
        # 3. nesterov...
        hint = jtu.tree_map(lambda ph, h: h + beta_3 * (ph - h), previous_hint, updates)
    elif method == 4:
        # 4. weighted Average, WA
        hint = jtu.tree_map(lambda ph, h: 0.5 * ph + 0.5 * h, previous_hint, updates)
    elif method == 5:
        # 5. Polynomial WA
        if count_inc is None:
            raise ValueError("count_inc is required for method 5")
        k = count_inc 
        hint = jtu.tree_map(lambda ph, h: (k / (k + 1)) * ph + (1 / (k + 1)) * h, previous_hint, updates)
    elif method == 7:
        # 7. Harmonic Mean (performs badly)
        hint = jtu.tree_map(lambda ph, h: 2 * ph * h / (ph + h), previous_hint, updates)
    elif method == 8:
        # 8. Root Mean Square (performs badly)
        hint = jtu.tree_map(lambda ph, h: (ph**2 + h**2)**0.5 / 2**0.5, previous_hint, updates)
    elif method == 9:
        # 9. Logarithmic Weighted Average
        hint = jtu.tree_map(lambda ph, h: (jnp.log(1 + ph) + jnp.log(1 + h)) / 2, previous_hint, updates)
    elif method == 10:
        # 10. tanh Combination
        hint = jtu.tree_map(lambda ph, h: jnp.tanh(ph) + 0.1 * jnp.log(1 + h), previous_hint, updates)
    elif method == 11:
        # 11. Standard Momentum
        hint = jtu.tree_map(lambda ph, h: ph + beta_3 * ph + (1 - beta_3) * h, previous_hint, updates)
    elif method == 12:
        # 12. 
        hint = jtu.tree_map(lambda ph, h: ph + beta_3 * (ph + (1 - beta_3) * h), previous_hint, updates)
    elif method == 13:
        # 13. 
        hint = jtu.tree_map(lambda ph, h: ph + (1 - beta_3) * (h - ph) + beta_3 * ph, previous_hint, updates)
    elif method == 14:
        # 14. Adaptive Momentum
        hint = jtu.tree_map(lambda ph, h: ph + (1 - beta_3) * (h / (1 - beta_3)), previous_hint, updates)
    elif method == 15:
        # 15. RMSProp
        hint = jtu.tree_map(lambda ph, h: ph + (1 - beta_3) * (h / (jnp.sqrt(ph) + 1e-8)), previous_hint, updates)
    elif method ==16:
        hint = jtu.tree_map(lambda ph, h: beta_3 * ph + (1 - beta_3) * h, prev_params, updates)
    elif method == 17:
        hint = jtu.tree_map(lambda ph, h: h + beta_3 * (ph - h), prev_params, updates)
    elif method == 18:
        hint = jtu.tree_map(lambda u: 0*u, updates)
    elif method == 19:
        #hint = prev_params
        hint = jtu.tree_map(lambda ph: ( ph / utils.tree_l2_norm(ph) ) * utils.tree_l2_norm(m_t), prev_params)

    elif method == 20:
        hint = jtu.tree_map(lambda ph: ph, prev_params)
    else:
        raise ValueError("Invalid method number. Must be between 0 and 20.")
    return hint


class OFTRLState(NamedTuple):
    count: chex.Array
    weighted_sum: chex.Array
    sum_squared: chex.Array
    prev_hint: chex.Array
    prev_updates: chex.Array
    prev_params: chex.Array
    logging: logstate.Log

def oftrl( #with cheating hint (no adaptive)
    lr: optax.ScalarOrSchedule,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    beta_3: float =0.9,
    epsilon: float = 1e-8,
    hint_method: int = 3,
    cheating: bool = False
) -> OnlineLearnerExtraArgs:

    class LogChecker(object):
        """A dummy class to make sure all logs have the same structure and data type."""
        def __init__(self):
            self.logging = {
                "hint/distance(prev_hint, grad)": jnp.zeros([]),
                "hint/distance(prev_hint, prev_grad)": jnp.zeros([]),
                "hint/<prev_hint, grad>": jnp.zeros([]),
                "hint/norm(prev_hint)": jnp.zeros([]),
                "grad/distance(prev_grad, grad)": jnp.zeros([]),
                "hint/cos(grad, prev_hint)": jnp.zeros([]),
                "hint/cos(hint, prev_hint)": jnp.zeros([]),
                "hint/distance(hint, prev_hint)": jnp.zeros([]),
                "hint/cos(hint, grad)": jnp.zeros([]),
                "hint/distance(hint, grad)": jnp.zeros([]),
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
        initial_weighted_sum = jtu.tree_map(jnp.zeros_like, params)
        initial_sum_squared = jtu.tree_map(jnp.zeros_like, params)
        initial_hint = jtu.tree_map(lambda x: jnp.zeros_like(x), params)        
        initial_updates = jtu.tree_map(lambda x: jnp.zeros_like(x), params)
        return OFTRLState(
            count=jnp.zeros([], jnp.int32),
            weighted_sum=initial_weighted_sum,
            sum_squared=initial_sum_squared,
            logging=logger(),
            prev_hint = initial_hint,
            prev_updates = initial_updates,
            prev_params=initial_updates
        )

    def update_fn(updates, state, params, hint):
        if callable(lr):
            eta = lr(state.count)
        else:
            eta = lr
        count_inc = optax.safe_int32_increment(state.count)


        previous_hint = state.prev_hint
        
        #method between 1 and 20
        if not cheating:
            hint = update_hint(hint_method, previous_hint, updates, count_inc, state.prev_params, state.weighted_sum, beta_3)
          
        
        #TODO: new_weighted_sum_hint and new_sum_squared_hint should only be calculated if hint is not zero hint (otherwise it's a useless calculation)
        new_weighted_sum = jtu.tree_map(lambda ws, u: beta_1 * ws + (1-beta_1) * u, state.weighted_sum, updates)
        new_weighted_sum_hint = jtu.tree_map(lambda nws, h: beta_1 * nws + (1-beta_1) * h, new_weighted_sum, hint)

        new_sum_squared = jtu.tree_map(lambda ss, u: beta_2 * ss + (1-beta_2)*(u ** 2), state.sum_squared, updates)
        new_sum_squared_hint = jtu.tree_map(lambda nss, h: beta_2 * nss + (1-beta_2)*(h**2), new_sum_squared, hint)

        is_zero_hint = jnp.all(jnp.array([jnp.all(h == 0) for h in jtu.tree_leaves(hint)]))
        params_next = jax.lax.cond( #with debiasing
            is_zero_hint,
            lambda _: jtu.tree_map(
                lambda ws, ss: (eta * (ws / (1 - beta_1 ** count_inc))) / (jnp.sqrt(ss / (1 - beta_2 ** count_inc) + epsilon)), 
                new_weighted_sum, new_sum_squared
            ),
            lambda _: jtu.tree_map(
                lambda ws, ss: (eta * (ws / (1 - beta_1 ** count_inc))) / (jnp.sqrt(ss / (1 - beta_2 ** count_inc) + epsilon)), 
                new_weighted_sum_hint, new_sum_squared_hint
            ),
            operand=None
        )

        params_next = jtu.tree_map(lambda s: -s, params_next)

        # logging
        hint_distance = utils.tree_l2_norm(jtu.tree_map(lambda x, y: x - y, previous_hint, updates))
        hint_distance_prev_updates = utils.tree_l2_norm(jtu.tree_map(lambda x, y: x - y, previous_hint, state.prev_updates))
        hint_inner_product = utils.tree_inner_product(previous_hint, updates)
        hint_norm = utils.tree_l2_norm(previous_hint)
        grad_distance = utils.tree_l2_norm(jtu.tree_map(lambda x, y: x - y, state.prev_updates, updates))

        new_state = OFTRLState(
            count=count_inc,
            weighted_sum=new_weighted_sum,
            sum_squared=new_sum_squared,
            prev_hint=hint,
            prev_updates=updates,
            prev_params=params_next,
            logging=logger(**{
                "hint/distance(prev_hint, grad)": hint_distance,
                "hint/distance(prev_hint, prev_grad)": hint_distance_prev_updates,
                "hint/<prev_hint, grad>": hint_inner_product,
                "hint/norm(prev_hint)": hint_norm,
                "grad/distance(prev_grad, grad)": grad_distance,
                "hint/cos(grad, prev_hint)": utils.tree_cosine_similarity(previous_hint, updates),
                "hint/cos(hint, prev_hint)": utils.tree_cosine_similarity(previous_hint, hint),
                "hint/distance(hint, prev_hint)": utils.tree_l2_norm(jtu.tree_map(lambda x, y: x - y,previous_hint, hint)),
                "hint/cos(hint, grad)": utils.tree_cosine_similarity(updates, hint),
                "hint/distance(hint, grad)": utils.tree_l2_norm(jtu.tree_map(lambda x, y: x - y,updates, hint))
            })
        )

        return params_next, new_state

    return OnlineLearnerExtraArgs(init_fn, update_fn)



class OFTRL2State(NamedTuple):
    count: chex.Array
    weighted_sum: chex.Array
    sum_squared: chex.Array
    prev_hint: chex.Array
    prev_updates: chex.Array
    logging: logstate.Log

def aggressive_oftrl(
    lr: optax.ScalarOrSchedule,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    beta_3: float =0.9,
    epsilon: float = 1e-8,
    hint_method: int = 3,
    c: float = 0.01,
    correlation: bool = False
) -> OnlineLearner:

    class LogChecker(object):
        """A dummy class to make sure all logs have the same structure and data type."""
        def __init__(self):
            self.logging = {
                "hint/distance(prev_hint, grad)": jnp.zeros([]),
                "hint/distance(prev_hint, prev_grad)": jnp.zeros([]),
                "hint/<prev_hint, grad>": jnp.zeros([]),
                "hint/norm(prev_hint)": jnp.zeros([]),
                "grad/distance(prev_grad, grad)": jnp.zeros([]),
                "hint/cos(grad, prev_hint)": jnp.zeros([]),
                "hint/cos(hint, prev_hint)": jnp.zeros([]),
                "hint/distance(hint, prev_hint)": jnp.zeros([]),
                "hint/cos(hint, grad)": jnp.zeros([]),
                "hint/distance(hint, grad)": jnp.zeros([]),
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
        initial_weighted_sum = jtu.tree_map(jnp.zeros_like, params)
        initial_sum_squared = jtu.tree_map(jnp.zeros_like, params)
        initial_hint = jtu.tree_map(lambda x: jnp.zeros_like(x), params)
        initial_updates = jtu.tree_map(lambda x: jnp.zeros_like(x), params)
        return OFTRL2State(
            count=jnp.zeros([], jnp.int32),
            weighted_sum=initial_weighted_sum,
            sum_squared=initial_sum_squared,
            logging=logger(),
            prev_hint = initial_hint,
            prev_updates=initial_updates,
            prev_params=initial_updates
        )

    def update_fn(updates, state, params, hint):
        if callable(lr):
            eta = lr(state.count)
        else:
            eta = lr
        count_inc = optax.safe_int32_increment(state.count)


        #if hint is None:
        #    jax.debug.print(f"hint is None, iteration: {count_inc}")
        #    hint = jtu.tree_map(lambda x: jnp.zeros_like(x), params)
       # if jtu.tree_all(jtu.tree_map(lambda h: jnp.all(h == 0), hint)):

        #    jax.debug.print(f"hint is zero, iteration: {count_inc}")

        
        previous_hint = state.prev_hint
        
        #next_hint = jtu.tree_map(lambda ph, h: 0.9 * ph + (1 - 0.9) * h, previous_hint, hint)
        #next_hint = jtu.tree_map(lambda x: jnp.zeros_like(x), params)
        #hint = jtu.tree_map(lambda u, nh, ph: (u + (ph - nh)), updates, hint, previous_hint)
        
        #method between 1 and 16
        hint = update_hint(hint_method, previous_hint, updates, count_inc, state.prev_params, beta_3)

        
        new_weighted_sum = jtu.tree_map(lambda ws, u: beta_1 * ws + (1 - beta_1) * u, state.weighted_sum, updates)
        new_weighted_sum_hint = jtu.tree_map(lambda nws, h: beta_1 * nws + (1 - beta_1) * h, new_weighted_sum, hint)

        # adaptive in recursive 
        if correlation:
            #new_sum_squared = jtu.tree_map(lambda ss, u, pu: beta_2 * ss + (1 - beta_2) * jnp.maximum((u - pu) ** 2 - u ** 2, c), state.sum_squared, updates, state.prev_updates)
            new_sum_squared = jtu.tree_map(lambda nss, h, u: beta_2 * nss + (1 - beta_2) * jnp.maximum((h - u) ** 2 - h ** 2, c), new_sum_squared, hint, updates)
        else:
            #new_sum_squared = jtu.tree_map(lambda ss, u, pu: beta_2 * ss + (1 - beta_2) * (jnp.linalg.norm(u-pu) ** 2),state.sum_squared, updates, state.prev_updates)
            new_sum_squared = jtu.tree_map(lambda nss, h, u: beta_2 * nss + (1-beta_2)*((h-u)**2), new_sum_squared, hint, updates) #coordinate wise squaring


        is_zero_hint = jnp.all(jnp.array([jnp.all(h == 0) for h in jtu.tree_leaves(hint)]))
        scale = jax.lax.cond(
            is_zero_hint,
            lambda _: jtu.tree_map(lambda ws, ss: (eta * ws) / (jnp.sqrt(ss + epsilon)), new_weighted_sum, new_sum_squared),
            lambda _: jtu.tree_map(lambda ws, ss: (eta * ws) / (jnp.sqrt(ss + epsilon)), new_weighted_sum_hint, new_sum_squared),
            operand=None
        )

        params_next = jtu.tree_map(lambda s: -s, scale)

        # logging
        hint_distance = utils.tree_l2_norm(jtu.tree_map(lambda x, y: x - y, previous_hint, updates))
        hint_distance_prev_updates = utils.tree_l2_norm(jtu.tree_map(lambda x, y: x - y, previous_hint, state.prev_updates))
        hint_inner_product = utils.tree_inner_product(previous_hint, updates)
        hint_norm = utils.tree_l2_norm(previous_hint)
        grad_distance = utils.tree_l2_norm(jtu.tree_map(lambda x, y: x - y, state.prev_updates, updates))

        new_state = OFTRL2State(
            count=count_inc,
            weighted_sum=new_weighted_sum,
            sum_squared=new_sum_squared,
            prev_hint=hint,
            prev_updates=updates,
            prev_params=params_next,
            logging=logger(**{
                "hint/distance(prev_hint, grad)": hint_distance,
                "hint/distance(prev_hint, prev_grad)": hint_distance_prev_updates,
                "hint/<prev_hint, grad>": hint_inner_product,
                "hint/norm(prev_hint)": hint_norm,
                "grad/distance(prev_grad, grad)": grad_distance,
                "hint/cos(grad, prev_hint)": utils.tree_cosine_similarity(previous_hint, updates),
                "hint/cos(hint, prev_hint)": utils.tree_cosine_similarity(previous_hint, hint),
                "hint/distance(hint, prev_hint)": utils.tree_l2_norm(jtu.tree_map(lambda x, y: x - y,previous_hint, hint)),
                "hint/cos(hint, grad)": utils.tree_cosine_similarity(updates, hint),
                "hint/distance(hint, grad)": utils.tree_l2_norm(jtu.tree_map(lambda x, y: x - y,updates, hint))
            })
        )

        return params_next, new_state

    return OnlineLearner(init_fn, update_fn)

""" 

class OFTRL3State(NamedTuple):
    count: chex.Array
    weighted_sum: chex.Array
    sum_squared: chex.Array
    prev_hint: chex.Array
    prev_updates: chex.Array
    logging: logstate.Log

def oftrl( #with cheating hint (with adaptive) (with )
    lr: optax.ScalarOrSchedule,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
) -> OnlineLearner:

    class LogChecker(object):
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

    def init_fn(params):
        initial_weighted_sum = jtu.tree_map(jnp.zeros_like, params)
        initial_sum_squared = jtu.tree_map(jnp.zeros_like, params)
        initial_hint = jtu.tree_map(lambda x: jnp.zeros_like(x), params)
        initial_updates = jtu.tree_map(lambda x: jnp.zeros_like(x), params)
        return OFTRL3State(
            count=jnp.zeros([], jnp.int32),
            weighted_sum=initial_weighted_sum,
            sum_squared=initial_sum_squared,
            logging=logger(),
            prev_hint = initial_hint,
            prev_updates=initial_updates,
        )

    def update_fn(updates, state, params, hint):
        if callable(lr):
            eta = lr(state.count)
        else:
            eta = lr
        count_inc = optax.safe_int32_increment(state.count)


        #if hint is None:
        #    jax.debug.print(f"hint is None, iteration: {count_inc}")
        #    hint = jtu.tree_map(lambda x: jnp.zeros_like(x), params)
       # if jtu.tree_all(jtu.tree_map(lambda h: jnp.all(h == 0), hint)):

        #    jax.debug.print(f"hint is zero, iteration: {count_inc}")

        
        previous_hint = state.prev_hint
        
        #next_hint = jtu.tree_map(lambda ph, h: 0.9 * ph + (1 - 0.9) * h, previous_hint, hint)
        #next_hint = jtu.tree_map(lambda x: jnp.zeros_like(x), params)
        #hint = jtu.tree_map(lambda u, nh, ph: (u + (ph - nh)), updates, hint, previous_hint)
        
        #method between 1 and 10
        hint = update_hint(3, previous_hint, updates, count_inc)
          
        
        new_weighted_sum = jtu.tree_map(lambda ws, u: beta_1 * ws + (1 - beta_1) * u, state.weighted_sum, updates)
        new_weighted_sum_hint = jtu.tree_map(lambda nws, h: beta_1 * nws + (1 - beta_1) * h, new_weighted_sum, hint)

        # adaptive in recursive 
        #new_sum_squared = jtu.tree_map(lambda ss, u, pu: beta_2 * ss + (1 - beta_2) * (jnp.linalg.norm(u-pu) ** 2),state.sum_squared, updates, state.prev_updates)
        #new_sum_squared_hint = jtu.tree_map(lambda nss, h, u: beta_2 * nss + (1-beta_2)*(jnp.linalg.norm(h-u)**2), new_sum_squared, hint, updates)
        
        #correlation instead of distance
        c = 0.01
        new_sum_squared = jtu.tree_map(lambda ss, u, pu: beta_2 * ss + (1 - beta_2) * jnp.maximum((u - pu) ** 2 - u ** 2, c), state.sum_squared, updates, state.prev_updates)
        new_sum_squared_hint = jtu.tree_map(lambda nss, h, u: beta_2 * nss + (1 - beta_2) * jnp.maximum((h - u) ** 2 - h ** 2, c), new_sum_squared, hint, updates)


        is_zero_hint = jnp.all(jnp.array([jnp.all(h == 0) for h in jtu.tree_leaves(hint)]))
        scale = jax.lax.cond(
            is_zero_hint,
            lambda _: jtu.tree_map(lambda ws, ss: (eta * ws) / (jnp.sqrt(ss + epsilon)), new_weighted_sum, new_sum_squared),
            lambda _: jtu.tree_map(lambda ws, ss: (eta * ws) / (jnp.sqrt(ss + epsilon)), new_weighted_sum_hint, new_sum_squared_hint),
            operand=None
        )

        params_next = jtu.tree_map(lambda s: -s, scale)

        # logging
        hint_distance = utils.tree_l2_norm(jtu.tree_map(lambda x, y: x - y, previous_hint, updates))
        hint_inner_product = utils.tree_inner_product(previous_hint, updates)

        new_state = OFTRL3State(
            count=count_inc,
            weighted_sum=new_weighted_sum,
            sum_squared=new_sum_squared,
            prev_hint=hint,
            prev_updates=updates,
            logging=logger(**{
                "hint/norm(prev_hint, grad)": hint_distance,
                "hint/<prev_hint, grad>": hint_inner_product
            })
        )

        return params_next, new_state

    return OnlineLearner(init_fn, update_fn)

 """