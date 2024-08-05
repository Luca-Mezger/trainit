import jax
from jax import numpy as jnp
from optax import Updates, Params
from typing import Any, Tuple, NamedTuple, Optional
from tqdm import tqdm
import sys
sys.path.append('../trainit')
import logstate
import utils
import wandb
from jax import numpy as jnp  
from typing import NamedTuple  # imports NamedTuple for creating custom tuple types
from optimizer.online_learners import OnlineLearnerExtraArgs



import jax.numpy as jnp
from jax import tree_map, grad
import jax

from typing import Any, Tuple, NamedTuple, Optional, Callable

from jax import numpy as jnp  
from jax import tree_util as jtu
import chex
import optax
from typing import NamedTuple
from optimizer.online_learners import OnlineLearner
import functools
from jax import numpy as jnp  
from jax import tree_util as jtu
import chex
import optax
from typing import NamedTuple
from optimizer.online_learners import OnlineLearner


from jax import numpy as jnp
from jax import grad
import chex
import optax
from typing import NamedTuple
from optimizer.online_learners import OnlineLearner

import jax
from jax import numpy as jnp
from jax import grad, jit
import chex
import optax
from typing import NamedTuple, Optional
from optimizer.online_learners import OnlineLearner, OnlineLearnerInitFn, OnlineLearnerUpdateFn

class AOOMDState(NamedTuple):
    count: chex.Array
    hint: chex.Array  # Store the previous hint
    v_t: chex.Array
    logging: logstate.Log  # Logging object to track metrics

def adaptive_oomd(
    D: float = 0.1,
    beta: float = 0.9,  # Momentue previous hint term for th
    beta_2: float = 0.9, #beta for lr calculation
) -> OnlineLearner:
    
    class LogChecker(object):
        """A dummy class to make sure all logs have the same structure and data type."""
        def __init__(self):
            self.logging = {
                "hint/norm(prev_hint, grad)": jnp.zeros([]),
                "hint/<prev_hint, grad>": jnp.zeros([]),
                "learning_rate/eta_t": jnp.zeros([]),
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
        #scalar learning rate
        #g_t = utils.tree_l2_norm(updates)
        #v_t_new = beta_2 * state.v_t + (1 - beta_2) * g_t ** 2
        #eta_t = D / (2 * jnp.sqrt(v_t_new))

        #current_grad = grad_mirror_map(params)
        #previous_hint = state.hint
        #next_hint = jtu.tree_map(lambda ph, h: beta * ph + (1 - beta) * h, previous_hint, updates) #h_{t+1} = β * h_t + (1 - β) * g_t
        # Update rule: x_{t+1} = x_t - learning_rate * (g_t + (h_t - h_{t+1}))
        #combined_updates = jtu.tree_map(lambda u, nh, ph: eta_t * (u + (ph - nh)), updates, next_hint, previous_hint)
        #dual_variable = jtu.tree_map(lambda cg, cu: (cg - cu), current_grad, combined_updates)
        #params_next = jtu.tree_map(inverse_grad_mirror_map, dual_variable)

        #coordinate wise
        g_t_squared = jtu.tree_map(lambda x: x ** 2, updates)
        v_scaled = jtu.tree_map(lambda v: utils.tree_scalar_multiply(v, beta_2), state.v_t)
        g_scaled = jtu.tree_map(lambda g: utils.tree_scalar_multiply(g, (1 - beta_2)), g_t_squared)
        v_t_new = jtu.tree_map(lambda v, g: v + g, v_scaled, g_scaled)
        eta_t = jtu.tree_map(lambda v: D / (2 * jnp.sqrt(1e-8 + v)), v_t_new)
 
        current_grad = grad_mirror_map(params)
        previous_hint = state.hint
        next_hint = jtu.tree_map(lambda ph, h: beta * ph + (1 - beta) * h, previous_hint, updates) #h_{t+1} = β * h_t + (1 - β) * g_t
        #Update rule: x_{t+1} = x_t - learning_rate * (g_t + (h_t - h_{t+1}))
        hint_differences = jtu.tree_map(lambda nh, ph: nh - ph, next_hint, previous_hint)
        updated_with_hints = jtu.tree_map(lambda u, hd: u + hd, updates, hint_differences)
        combined_updates = jtu.tree_map(lambda eta, uwh: eta * uwh, eta_t, updated_with_hints)
        dual_variable = jtu.tree_map(lambda cg, cu: (cg - cu), current_grad, combined_updates)
        params_next = jtu.tree_map(inverse_grad_mirror_map, dual_variable)
        
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
                "learning_rate/eta_t": eta_t,
            })
        )

    return OnlineLearner(init_fn, update_fn)



def train_step(learner, loss_fn, params, opt_state):
    grads = jax.grad(loss_fn)(params)
    params, opt_state = learner.update(grads, opt_state, params)
    return params, opt_state


def train(learner, loss_fn, params, num_steps):
    opt_state = learner.init(params)
    pbar = tqdm(range(num_steps), total=num_steps)
    for step in pbar:
        params, opt_state = train_step(learner, loss_fn, params, opt_state)
        loss = loss_fn(params)
        pbar.set_description(f"Step {step}, Params: {params}, Loss: {loss:.2f}")
        wandb.log({
            "params/norm": utils.tree_norm(params),
            "loss": loss,
        })

# simple quadratic loss
def simple_loss(params):
    target = jnp.array([1.0, -1.0, 0.5])  # Example target
    return jnp.sum((params - target) ** 2)


initial_params = jnp.array([0.0, 0.0, 0.0])

learning_rate = 0.1  # Or use optax.linear_schedule(init_value=0.1, end_value=0.01, transition_steps=100)
omd_learner = adaptive_oomd()
num_steps = 100

wandb.init(project="log1")

train(omd_learner, simple_loss, initial_params, num_steps)
