"""Secure-side 4-block BERT (U-split) with the round-8 MPC-friendly kernels, plus exact
float reference and the client head. Kernels carry analytic custom JVPs so the tangent
pass reuses the primal kernel output (exp' = exp, (1/(1+e))' = -y^2, sigma' = s(1-s))."""
import numpy as np
import jax, jax.numpy as jnp
from jax import lax

C_INV = [0.03226120897959244, -0.049780065730235586, 0.057650254381631766, -0.08583010236400039,
         0.1317100845857373, -0.19765769464931107, 0.29629765286612486, -0.44444230412031277,
         0.6666666475671728]  # 1/(1+e) on e in [0,1], centered at t=e-0.5, Horner high->low

@jax.custom_jvp
def exp_neg(x):
    """exp(x) for x in [-12,0]: 8-term Taylor of x/16, then 4 squarings."""
    z = jnp.clip(x, -12.0, 0.0) / 16.0
    acc = 1.0
    for k in range(8, 0, -1):
        acc = 1.0 + z * acc / k
    for _ in range(4):
        acc = acc * acc
    return acc

@exp_neg.defjvp
def _exp_neg_jvp(p, t):
    y = exp_neg(p[0]); return y, y * t[0]

@jax.custom_jvp
def inv_1p(e):
    t = e - 0.5
    acc = C_INV[0]
    for c in C_INV[1:]:
        acc = acc * t + c
    return acc

@inv_1p.defjvp
def _inv_1p_jvp(p, t):
    y = inv_1p(p[0]); return y, -y * y * t[0]

@jax.custom_jvp
def sigmoid(s):
    pos = s >= 0
    a = jnp.where(pos, s, -s)
    q = inv_1p(exp_neg(-a))
    return jnp.where(pos, q, 1.0 - q)

@sigmoid.defjvp
def _sig_jvp(p, t):
    y = sigmoid(p[0]); return y, y * (1.0 - y) * t[0]

@jax.custom_jvp
def gelu_mpc(z):
    s = 1.5957691216057308 * (z + (0.044715 * z * z) * z)   # 2*sqrt(2/pi)*(z+0.044715 z^3)
    return z * sigmoid(s)

@gelu_mpc.defjvp
def _gelu_jvp(pr, tg):
    # gelu' = sig(s) + z * sig'(s) * s'(z),  sig' = e q^2 with e = exp(-|s|), q = 1/(1+e).
    # sig' is zeroed for |s| >= 12: there its true value is < 6e-6 but under fixed point it is
    # rounding noise, and s'(z) ~ z^2 for BERT's outlier activations (|z| ~ 50) amplifies it.
    z, = pr; zt, = tg
    s = 1.5957691216057308 * (z + (0.044715 * z * z) * z)
    pos = s >= 0
    a = jnp.where(pos, s, -s)
    e = exp_neg(-a)
    q = inv_1p(e)
    y = jnp.where(pos, q, 1.0 - q)
    dsig = jnp.where(a < 12.0, e * q * q, 0.0)
    ds = 1.5957691216057308 * (1.0 + (0.134145 * z) * z)
    return z * y, (y + (z * dsig) * ds) * zt

@jax.custom_jvp
def _softmax_core(s, mask):
    m = jnp.max(s, axis=-1, keepdims=True)
    e = exp_neg(s - m) * mask
    return e * (1.0 / jnp.sum(e, axis=-1, keepdims=True))

@_softmax_core.defjvp
def _softmax_core_jvp(pr, tg):
    s, mask = pr; st, _ = tg
    p = _softmax_core(s, mask)
    return p, p * (st - jnp.sum(p * st, axis=-1, keepdims=True))

def softmax_mpc(s, mask):
    return _softmax_core(s + (mask - 1.0) * 30.0, mask)

LN_PRE = [64.0]

@jax.custom_jvp
def _norm_core(x):
    """x_hat = (x-mean)/std with the round-8 prescale S (protects c^2 against ring overflow)."""
    S = LN_PRE[0]
    d = (x - jnp.mean(x, axis=-1, keepdims=True)) / S
    return d * lax.rsqrt(jnp.mean(d * d, axis=-1, keepdims=True) + 1e-12 / (S * S))

@_norm_core.defjvp
def _norm_core_jvp(pr, tg):
    # analytic: d x_hat = r * (dd - x_hat * mean(x_hat * dd)),  dd = centered(dx)/S, r = 1/std(d)
    # keeps every factor O(1); differentiating rsqrt instead needs r^3 * mean(d dd), which loses
    # all precision under fixed point (r^3 huge, mean(d dd) tiny).
    x, = pr; xt, = tg
    S = LN_PRE[0]
    d = (x - jnp.mean(x, axis=-1, keepdims=True)) / S
    r = lax.rsqrt(jnp.mean(d * d, axis=-1, keepdims=True) + 1e-12 / (S * S))
    n = d * r
    dd = (xt - jnp.mean(xt, axis=-1, keepdims=True)) / S
    return n, r * (dd - n * jnp.mean(n * dd, axis=-1, keepdims=True))

def ln_mpc(x, g, b, eps=1e-12):
    return _norm_core(x) * g + b

def softmax_exact(s, mask):
    return jax.nn.softmax(jnp.where(mask > 0, s, -1e30), axis=-1)

def ln_exact(x, g, b, eps=1e-12):
    c = x - jnp.mean(x, axis=-1, keepdims=True)
    return c / jnp.sqrt(jnp.mean(c * c, axis=-1, keepdims=True) + eps) * g + b

def gelu_exact(z):
    return jax.nn.gelu(z, approximate=False)

KER = dict(mpc=(softmax_mpc, ln_mpc, gelu_mpc), exact=(softmax_exact, ln_exact, gelu_exact))

def encoder(h0, W, theta, mask, pos, kind='mpc', trim=False, last_row=False, taps=False):
    """h0 [1,T,256]; theta: 16 arrays (layer, Q/V, A[4,256]/B[256,4]); returns h at MASK [256]."""
    sm, ln, ge = KER[kind]
    x = h0
    if trim:  # public valid length (standard in secure inference); padded keys are exactly masked anyway
        n = int(mask.sum()); x = x[:, :n]; mask = mask[:n]
    T = x.shape[1]
    mk = mask[None, None, None, :]
    tp = []
    for i in range(4):
        Aq, Bq, Av, Bv = theta[4 * i: 4 * i + 4]
        w = lambda k: W[f'w{i}_{k}']
        q = x @ w('q') + w('qb') + ((x @ Aq.T) @ Bq.T) * 4.0
        k = x @ w('k') + w('kb')
        v = x @ w('v') + w('vb') + ((x @ Av.T) @ Bv.T) * 4.0
        if last_row and i == 3:  # only the MASK row is ever revealed: query side on 1 token
            x = x[:, pos:pos + 1]; q = q[:, pos:pos + 1]; Tq = 1
        else:
            Tq = T
        q = q.reshape(1, Tq, 4, 64).transpose(0, 2, 1, 3)
        k, v = [a.reshape(1, T, 4, 64).transpose(0, 2, 1, 3) for a in (k, v)]
        p = sm(q @ k.transpose(0, 1, 3, 2), mk)
        att = (p @ v).transpose(0, 2, 1, 3).reshape(1, Tq, 256)
        pre1 = x + att @ w('o') + w('ob')
        h = ln(pre1, w('ln1'), w('ln1b'))
        ff = ge(h @ w('ff1') + w('ff1b'))
        tp += [p, pre1, h, ff]
        pre2 = h + ff @ w('ff2') + w('ff2b')
        x = ln(pre2, w('ln2'), w('ln2b'))
        tp += [pre2, x]
    o = x[0, 0] if (last_row) else x[0, pos]
    return (o, tp) if taps else o

def client_loss(h, H, y):
    """Plaintext client: real MLM transform + 2 label-word rows + CE. FP64."""
    z = gelu_exact(h @ H['head_W'] + H['head_b'])
    z = ln_exact(z, H['head_g'], H['head_be'])
    logits = z @ H['head_dec'] + H['head_decb']
    return -jax.nn.log_softmax(logits)[y]

def load(path):
    F = dict(np.load(path))
    W = {k: v for k, v in F.items() if k.startswith('w')}
    H = {k: v for k, v in F.items() if k.startswith('head')}
    theta = [F[f'theta{j}'] for j in range(16)]
    nd = len([k for k in F if k.startswith('u') and k.endswith('_0')])
    us = [[F[f'u{d}_{j}'] for j in range(16)] for d in range(nd)]
    return F, W, H, theta, us
