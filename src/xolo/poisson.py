import jax
import jax.numpy as jnp

import tensorflow as tf

epsilon = 1e-5  # protect against small values in log

def l2_norm(params, alpha = 1.):
    return alpha * jnp.sum (jnp.array ([jnp.sum(x*x) for x in jax.tree_util.tree_leaves(params)]))

def poisson_loss (y_pred, y_true, epsilon=epsilon):
    # Shape of y_pred and y_true is (batches,bins,features)
    assert y_pred.shape == y_true.shape, f"predicted shape {y_pred.shape} != true shape {y_true.shape}"
    # Poisson loss is negative log-likelihood assuming y_true ~ Poisson(y_pred):
    # L = -log(e^{-y_pred} y_pred^{y_true} / (y_true!))
    #   = y_pred - y_true*log(y_pred) + log(y_true!)
    #   = y_pred - y_true*log(y_pred) + (y_true * log(y_true) - y_true)       [Stirling]
    #   = y_pred + y_true*(log(y_true) - log(y_pred) - 1)
    # If we retain only the y_pred-dependent terms we get the Keras Poisson loss:
    # L = y_pred - y_true*log(y_pred)  + terms independent of y_pred
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/Poisson
    y_pred = y_pred + epsilon
    y_true = y_true + epsilon
    return jnp.mean (y_pred - y_true * jnp.log(y_pred))

def apply_poisson_loss (params, f, x, y_true, epsilon=epsilon, **kwargs):
    return poisson_loss (f(params,x,**kwargs), y_true, epsilon=epsilon)

def poisson_multinomial_loss (y_pred, y_true, epsilon=epsilon, total_weight=1., rescale=False):
    # Combines a Poisson loss for total # of features with a (weighted) multinomial specificity term.
    # Includes baskerville's optional rescale term.
    seq_len = y_pred.shape[-2]
    y_pred = y_pred + epsilon  # (B,L,F)
    y_true = y_true + epsilon  # (B,L,F)
    s_pred = jnp.sum (y_pred, axis=-2, keepdims=True)  # (B,1,F)
    s_true = jnp.sum (y_true, axis=-2, keepdims=True)  # (B,1,F)
    p_loss = poisson_loss (s_pred, s_true, epsilon=0.) / seq_len
    m_loss = -jnp.mean (y_true * jnp.log(y_pred / s_pred))
    return (m_loss + total_weight*p_loss) * jnp.where (rescale, 2/(1+total_weight), 1)

def weighted_poisson_loss (y_pred, y_true, weight, epsilon=epsilon):
    y_pred = y_pred + epsilon
    y_true = y_true + epsilon
    loss = y_pred - y_true * jnp.log(y_pred)
    weighted_loss = weight * loss
#    jax.debug.print ("y_pred {} sum {} min {} max {}", y_pred, jnp.sum(y_pred), jnp.min(y_pred), jnp.max(y_pred))
#    jax.debug.print ("y_true {} sum {} min {} max {}", y_true, jnp.sum(y_true), jnp.min(y_true), jnp.max(y_true))
#    jax.debug.print ("loss {} sum {}", loss, jnp.sum(loss))
#    jax.debug.print ("weighted_loss {} sum {}", weighted_loss, jnp.sum(weighted_loss))
#    jax.debug.print ("weight {} sum {}", weight, jnp.sum(weight))
    return jnp.mean (weight * (y_pred - y_true * jnp.log(y_pred)))

def compute_xy_moments (x, y, weights=None, warn_if_zero=True):   # conventionally x=pred, y=true
    assert x.shape == y.shape, f"shape of predicted values {x.shape} != shape of true values {y.shape}"
    if x.ndim > 1:
        axis = tuple(range(x.ndim - 1))  # shape of x,y is (batches, bins, features); we want to keep features
    else:
        axis = None  # x is a single vector, so we assume it has been flattened (e.g. a sparse matrix of exon-exon junction coords) and just sum over everything
    if weights is not None:
        x = x * weights
        y = y * weights
        n = jnp.ones(x.shape[-1]) * jnp.sum(weights,axis=axis)
    elif x.ndim > 1:
        n = jnp.ones(x.shape[-1]) * jnp.prod(jnp.array(x.shape[:-1]))
    else:
        n = x.shape[0]
    ex, ey, exx, eyy, exy = jnp.sum(x,axis=axis), jnp.sum(y,axis=axis), jnp.sum(x*x,axis=axis), jnp.sum(y*y,axis=axis), jnp.sum(x*y,axis=axis)
    xv = exx - ex*ex
    yv = eyy - ey*ey
    def dummy_print():
        pass
    def debug_print():
        jax.debug.print("WARNING: compute_xy_moments: zero variation, xv={} yv={}\n\nexx:\n{}\n\nex:\n{}\n\neyy:\n{}\n\ney:\n{}\n\n",
                        xv, yv, exx, ex, eyy, ey)
    if warn_if_zero:  # turned off by default otherwise this is slow and annoying
        jax.lax.cond (jnp.min(yv) == 0, debug_print, dummy_print)
    result = jnp.array ((n, ex, ey, exx, eyy, exy))  # (6, features)
    result = result.transpose()  # (features, 6)
    return result  # Moments are indexed as (n, xSum, ySum, xxSum, yySum, xySum)

def validate_xy_moments_shape (xy_moments):
    assert xy_moments.shape[-1]==6, f"expected final dimension of inputs to be 6 (n, xSum, ySum, xxSum, yySum, xySum): {xy_moments.shape}"

def sum_xy_moments (xy_moments):
    validate_xy_moments_shape (xy_moments)
    return jnp.sum (xy_moments, axis=tuple(range(xy_moments.ndim - 1)))

def zero_xy_moments (features: int = 1):
    return jnp.zeros ((features, 6))  # Moments are indexed as (n, xSum, ySum, xxSum, yySum, xySum)

# Pseudocounts to avoid division by zero errors
def epsilon_where_zero (x):
    return jnp.where (x==0, epsilon, 0)

def safe_divide (x, y):
    pc = epsilon_where_zero (y)
    return x / (y + pc)

# Pearson correlation coefficient
def pearson_r (xy_moments, keep_features = False):
    validate_xy_moments_shape (xy_moments)
    if not keep_features:
        xy_moments = sum_xy_moments (xy_moments)
    n, sx, sy, sxx, syy, sxy = tuple (xy_moments.transpose())
    (ex, ey, exx, eyy, exy) = [q/n for q in (sx, sy, sxx, syy, sxy)]
    return safe_divide (exy - ex * ey, jnp.sqrt ((exx - ex ** 2) * (eyy - ey ** 2)))

# Coefficient of determination. Note this is NOT symmetric to {x,y} permutation. Assumes x=pred, y=true
def r_squared (xy_moments, keep_features = False):
    validate_xy_moments_shape (xy_moments)
    if not keep_features:
        xy_moments = sum_xy_moments (xy_moments)
    n, _sx, sy, sxx, syy, sxy = tuple (xy_moments.transpose())
    ey = sy / n
    ss_res = sxx + syy - 2 * sxy
    ss_tot = syy - ey * sy
    return 1 - safe_divide (ss_res, ss_tot)