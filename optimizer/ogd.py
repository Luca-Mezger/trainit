from jax import numpy as jnp  
from jax import tree_util as jtu  # imports tree utilities from jax for operations on pytrees
import chex  # imports chex for JAX type and shape checking
import optax  # imports optax for optimization tools
from typing import NamedTuple, Optional  # imports NamedTuple for creating custom tuple types
import logstate
from optimizer.online_learners import OnlineLearner
import utils


class OGDState(NamedTuple):  
    count: chex.Array  

def temp_ogd(
    learning_rate: optax.ScalarOrSchedule, 
    beta: float = 0.99,
    mu: float = 100.0,
    ) -> OnlineLearner:  # returns OnlineLearner object
    """Online Gradient Descent (OGD) with weight decay and momentum constants.

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
        adjusted_updates = jtu.tree_map(lambda u: u * (1 + lr * mu), ogd_updates)
        new_params = jtu.tree_map(lambda p, u: beta * (p + u), params, adjusted_updates)



        count_inc = optax.safe_int32_increment(state.count)  # safely increments the count (avoids overflow)
        #use ogd_updates as return value = SGD
        return new_params, OGDState(count=count_inc)  # returns OGD updates and the new state
    
    return OnlineLearner(init_fn, update_fn) 


class AOGDState(NamedTuple):  
    count: chex.Array  
    v_t: chex.Array

def _ogd(
    learning_rate: optax.ScalarOrSchedule, 
    beta: float = 0.9,
    beta_2: float = 0.9,
    ) -> OnlineLearner:  # returns OnlineLearner object
    """Online Gradient Descent (OGD) with weight decay and momentum constants.

    Args:
        learning_rate: learning rate for the gradient descent
    """

    def init_fn(params=None):  # initializes the optimizer state
        del params  # deletes params
        return AOGDState(count=jnp.zeros([], jnp.int32), v_t=jnp.zeros([], jnp.float32))  # returns OGDState with count set to zero
    
    def update_fn(updates, state, params):  # define update function

        g_t = utils.tree_l2_norm(updates)
        v_t_new = 0.9 * state.v_t + (1 - 0.9) * g_t ** 2
        # Update rule for eta_t
        eta_t = 0.1 / (2 * jnp.sqrt(v_t_new))


        ogd_updates = jtu.tree_map(lambda g: -eta_t * g, updates)  # computes the OGD updates
        #adjusted_updates = jtu.tree_map(lambda u: u * (1 + lr * mu), ogd_updates)
        new_params = jtu.tree_map(lambda p, u: p + u, params, ogd_updates)



        count_inc = optax.safe_int32_increment(state.count)  # safely increments the count (avoids overflow)
        #use ogd_updates as return value = SGD
        return new_params, AOGDState(count=count_inc, v_t=v_t_new)  # returns OGD updates and the new state
    
    return OnlineLearner(init_fn, update_fn) 





#----------------Doesn't work yet--------
#instead of changing pipeline in train_jax.py --> pass y_t as new_param and store x_t in SOGDState, then "umformen" nach x_t --> ask Qinchzi :)
class SOGDState(NamedTuple):
    z: chex.Array  
    count: chex.Array
    custom_gradient_point: chex.Array  # Add this field to store the custom point
def sfogd( #schedule_free_ogd
    learning_rate: optax.ScalarOrSchedule, 
    beta: float = 0.9,
    ) -> OnlineLearner:  # returns OnlineLearner object
    """Schedule-Free Stochastic Gradient Descent (OGD) https://arxiv.org/pdf/2405.15682

    Args:
        learning_rate: learning rate for the gradient descent
        beta: momentum parameter interpolating between updates
    """

    def init_fn(params=None):  
        return SOGDState(z=params, count=jnp.zeros([], jnp.int32), custom_gradient_point=params)  # Initializes z with params and count to zero

    def update_fn(updates, state, params):  
        z, count = state.z, state.count
        
        if callable(learning_rate):  
            lr = learning_rate(count)  # get learning rate if it's a schedule (uhhh...)
        else:
            lr = learning_rate

        # Update rules as per Schedule-Free SGD
        gradient = jtu.tree_map(lambda g: -lr * g, updates)  # Apply learning rate to the gradients
        z_next = jtu.tree_map(lambda z, g: z + g, z, gradient)  # Update z

        c_t1 = 1 / (count + 1)  # Calculate c_{t+1}
        new_params = jtu.tree_map(lambda x, z: (1 - c_t1) * x + c_t1 * z, params, z_next)  # Update x_t

        y_t = jtu.tree_map(lambda z, x: (1 - beta) * z + beta * x, z_next, new_params)  # Calculate y_t
        count_inc = optax.safe_int32_increment(count)  # Increment the count
        return new_params, SOGDState(z=z_next, count=count_inc, custom_gradient_point=y_t)  # Return new params and updated state

    return OnlineLearner(init_fn, update_fn)



class ASGDState(NamedTuple):
    count: jnp.array  # iteration counter
    sum_squared_grads: jnp.array  # Sum of squared gradients
    logging: logstate.Log

def aogd(learning_rate: float, epsilon: float=1e-4, beta=1) -> OnlineLearner:
    class LogChecker(object):
        """A dummy class to make sure all logs have the same structure and data type."""
        def __init__(self):
            self.logging = {
                "learning_rate/norm(lr)": jnp.zeros([]),
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
        # Initialize the sum of squared gradients and count
        sum_squared_grads = jnp.zeros([])
        count = jnp.zeros([], jnp.int32)
        return ASGDState(count=count, sum_squared_grads=sum_squared_grads,logging=logger(),)

    def update_fn(grads, state, params):

        squared_norm = utils.tree_squared_l2_norm(grads)
        total_squared_grads = state.sum_squared_grads + squared_norm
        adaptive_lr = learning_rate / jnp.sqrt(epsilon + total_squared_grads)
        
        # Update parameters
        updates = jtu.tree_map(lambda g: -adaptive_lr * g, grads)
        new_params = jtu.tree_map(lambda p, u: (p + u), params, updates)
        count_inc = optax.safe_int32_increment(state.count)

        
        return new_params, ASGDState(count=count_inc, sum_squared_grads=total_squared_grads, logging=logger(**{
                "learning_rate/norm(lr)": utils.tree_l2_norm(adaptive_lr)
            }))

    return OnlineLearner(init_fn, update_fn)



class AdagradState(NamedTuple):
    count: jnp.array  # Iteration counter
    sum_squared_grads: jnp.array  # Sum of squared gradients
    logging: logstate.Log

def ogd(beta, learning_rate: float, epsilon: float=1e-8) -> OnlineLearner:
    class LogChecker(object):
        """A dummy class to ensure all logs have consistent structure and data type."""
        def __init__(self):
            self.logging = {
                "learning_rate/norm(lr)": jnp.zeros([]),
                "learning_rate/avg(lr)": jnp.zeros([]),
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
        # Initialize the sum of squared gradients and count
        sum_squared_grads = jtu.tree_map(lambda x: jnp.zeros_like(x), params) if params is not None else None
        count = jnp.zeros([], jnp.int32)
        return AdagradState(count=count, sum_squared_grads=sum_squared_grads, logging=logger(),)

    def update_fn(grads, state, params):
        # Calculate new sum of squared gradients
        new_sum_squared_grads = jtu.tree_map(lambda g, ssg: ssg + jnp.square(g), grads, state.sum_squared_grads)
        # Calculate adaptive learning rate
        adaptive_lr = jtu.tree_map(lambda ssg: learning_rate / (jnp.sqrt(ssg + epsilon)), new_sum_squared_grads)
        
        # Update parameters
        updates = jtu.tree_map(lambda lr, g: -lr * g, adaptive_lr, grads)
        new_params = jtu.tree_map(lambda p, u: 0.9*(p + u), params, updates)
        count_inc = optax.safe_int32_increment(state.count)

        return new_params, AdagradState(count=count_inc, sum_squared_grads=new_sum_squared_grads, logging=logger(**{
                "learning_rate/norm(lr)": utils.tree_l2_norm(adaptive_lr),
                "learning_rate/avg(lr)": jnp.mean(jnp.concatenate(jtu.tree_leaves(jtu.tree_map(jnp.ravel, adaptive_lr)))),
            }))

    return OnlineLearner(init_fn, update_fn)
