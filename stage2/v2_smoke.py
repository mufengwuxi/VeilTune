import torch, crypten, crypten.mpc as mpc, sys
sys.path.insert(0, '/workspace/veiltune')
@mpc.run_multiprocess(world_size=2)
def run():
    import gpu_crypten as G, gpu_crypten_v2 as V
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    out = {}
    def chk(name, fn, ref):
        try:
            got = fn(); err = float((got - ref).abs().max()); out[name] = f'ok maxerr={err:.2e}'
        except Exception as e:
            out[name] = f'FAIL {type(e).__name__}: {str(e)[:160]}'
    x = torch.randn(5, 7, dtype=torch.float64).cuda(); y = torch.randn(5, 7, dtype=torch.float64).cuda()
    X, Y = crypten.cryptensor(x, device='cuda'), crypten.cryptensor(y, device='cuda')
    chk('stack*unsqueeze', lambda: (crypten.stack([X, Y]) * X.unsqueeze(0)).get_plain_text(), torch.stack([x, y]) * x)
    chk('index_select', lambda: X.index_select(0, torch.tensor([4, 0, 2], device='cuda')).get_plain_text(), x[[4, 0, 2]])
    g = (X < 0).to(V.ARITH)
    chk('gate*secret', lambda: (g * Y).get_plain_text(), (x < 0).double() * y)
    chk('neg1p(gate)', lambda: V.neg1p(g).get_plain_text(), 1.0 - (x < 0).double())
    chk('to_fx*secret', lambda: (V.to_fx(g, Y) * Y).get_plain_text(), (x < 0).double() * y)
    chk('shift -3', lambda: V.shift(X, -3).get_plain_text(), x / 8)
    chk('shift +2', lambda: V.shift(X, 2).get_plain_text(), x * 4)
    chk('cmp_lt_batch', lambda: V.cmp_lt_batch(X, [0.0, 0.5]).get_plain_text(), torch.stack([(x < 0).double(), (x < 0.5).double()]))
    v = torch.exp(torch.randn(6, 1, dtype=torch.float64) * 3).cuda()
    ops = V.SecureOpsV2({k: torch.zeros(1).numpy() for k in ['w0_q']})
    chk('rsqrt', lambda: ops.rsqrt_newton(crypten.cryptensor(v, device='cuda')).get_plain_text(), v.rsqrt())
    d = torch.rand(6, 1, dtype=torch.float64).cuda() * 30 + 1
    chk('recip_den', lambda: ops.recip_den(crypten.cryptensor(d, device='cuda'), 38).get_plain_text(), 1 / d)
    xe = -torch.rand(6, 7, dtype=torch.float64).cuda() * 14
    chk('exp_neg', lambda: ops.exp_neg(crypten.cryptensor(xe, device='cuda'))[0].get_plain_text(), torch.exp(xe.clamp(min=-12)))
    z = torch.randn(6, 7, dtype=torch.float64).cuda() * 3
    gl = lambda t: t * 0.5 * (1 + torch.erf(t / 2 ** .5))
    chk('gelu', lambda: ops.gelu(crypten.cryptensor(z, device='cuda'))[0].get_plain_text(), gl(z))
    return out
if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    r = [x for x in run() if x][0]
    for k, v in r.items(): print(f'{k:16s} {v}')
