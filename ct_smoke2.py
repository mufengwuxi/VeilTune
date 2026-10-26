"""Probe MPCTensor API surface needed for the encoder."""
import torch
import crypten

@crypten.mpc.run_multiprocess(world_size=2)
def smoke():
    crypten.init()
    x = (torch.rand(2, 25, 25, device='cuda') * 10 - 5)
    ct = crypten.cryptensor(x, device='cuda')
    r = {}
    try:
        m = ct.max(dim=-1, keepdim=True)
        r['max'] = bool(torch.allclose(m.get_plain_text(), x.max(-1, keepdim=True).values, atol=1e-2))
    except Exception as e:
        r['max'] = repr(e)[:60]
    try:
        v = (ct * ct).sum(-1, keepdim=True) / 256
        s = v.sqrt().get_plain_text()
        r['sqrt'] = bool(torch.allclose(s, (x * x).sum(-1, keepdim=True).div(256).sqrt(), rtol=5e-3, atol=1e-3))
    except Exception as e:
        r['sqrt'] = repr(e)[:60]
    try:
        rs = v.rsqrt().get_plain_text()
        r['rsqrt'] = bool(torch.allclose(rs, (x * x).sum(-1, keepdim=True).div(256).rsqrt(), rtol=5e-2, atol=1e-3))
    except Exception as e:
        r['rsqrt'] = repr(e)[:60]
    try:
        w = (ct >= 0).where(ct, -ct)
        r['where'] = bool(torch.allclose(w.get_plain_text(), x.abs(), atol=1e-2))
    except Exception as e:
        r['where'] = repr(e)[:60]
    try:
        pub = torch.randn(25, 8, device='cuda')
        y = ct @ pub
        r['matmul_public'] = bool(torch.allclose(y.get_plain_text(), x @ pub, atol=1e-2))
    except Exception as e:
        r['matmul_public'] = repr(e)[:60]
    print('API_PROBE', r)

smoke()
