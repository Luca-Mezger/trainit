import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Union, Callable, Protocol
from tqdm import tqdm
import sys
sys.path.append('../trainit')
import utils
import online_learners as ol
import wandb
from jax import numpy as jnp  
from jax import tree_util as jtu  # imports tree utilities from jax for operations on pytrees
import chex  # imports chex for JAX type and shape checking
import optax  # imports optax for optimization tools
from typing import NamedTuple  # imports NamedTuple for creating custom tuple types
from optimizer.online_learners import OnlineLearner

class OGDState(NamedTuple):  
    count: chex.Array  # stores count as an array

def ogd(
    learning_rate: optax.ScalarOrSchedule, 
    ) -> OnlineLearner:  # returns OnlineLearner object
    """Online Gradient Descent (OGD).

    Args:
        learning_rate: learning rate for the gradient descent
    """

    def init_fn(params=None):  # initializes the optimizer state
        del params  # deletes params
        return OGDState(count=jnp.zeros([], jnp.int32))  # returns OGDState with count set to zero
    
    def update_fn(updates, state, params):  # define update function
        if callable(learning_rate): #is it callable (schedule) or scalar
            lr = learning_rate(state.count)  # get learning rate if it's a schedule
        else:
            lr = learning_rate
        
        ogd_updates = jtu.tree_map(lambda g: -lr * g, updates)  # computes the OGD updates
        new_params = jtu.tree_map(lambda p, u: p + u, params, ogd_updates)
        count_inc = optax.safe_int32_increment(state.count)  # safely increments the count (avoids overflow)
        #use ogd_updates as return value = SGD
        return new_params, OGDState(count=count_inc)  # returns OGD updates and the new state
    
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

learning_rate = 0.01  # Or use optax.linear_schedule(init_value=0.1, end_value=0.01, transition_steps=100)
ogd_learner = ogd(learning_rate=learning_rate)
num_steps = 200

wandb.init(project="log1")

train(ogd_learner, simple_loss, initial_params, num_steps)
