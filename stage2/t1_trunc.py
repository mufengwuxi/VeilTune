"""T1: CrypTen 2PC truncation bias (FM64, f=16). Errors measured in ulp against the exact
product of the fixed-point encodings: err = Z_int - X_int*Y_int/2^16."""
import json, sys, torch, numpy as np, crypten, crypten.mpc as mpc
N = 200_000
S = 2 ** 16

def stats(e):
    e = e.double().flatten()
    vals, cnt = torch.unique(torch.round(e * 1e6) / 1e6, return_counts=True)
    top = sorted(zip(cnt.tolist(), vals.tolist()), reverse=True)[:6]
    return dict(mean=float(e.mean()), std=float(e.std()), min=float(e.min()), max=float(e.max()),
                frac_abs_gt_2=float((e.abs() > 2).float().mean()),
                se_mean=float(e.std() / e.numel() ** 0.5), top_values=[[v, c] for c, v in top])

@mpc.run_multiprocess(world_size=2)
def run(dev):
    crypten.init()
    g = torch.Generator().manual_seed(7)
    cases = {}
    # (a) secret x secret, generic N(0,1) x N(0,1)
    x = torch.randn(N, generator=g, dtype=torch.float64); y = torch.randn(N, generator=g, dtype=torch.float64)
    # (b) secret x secret, update-shaped: lr*c (lr=3e-4, c~N(0,3)) x u~N(0,1)  (emu: q(q(lr*c)*u), lr*c secret)
    lc = 3e-4 * 3.0 * torch.randn(N, generator=g, dtype=torch.float64); u = torch.randn(N, generator=g, dtype=torch.float64)
    # (c) secret x public fraction: u secret, lr*c public (one public scalar per step -> use many scalars)
    for name, (a, b, pub) in dict(ss_generic=(x, y, False), ss_update=(u, lc, False), sp_update=(u, lc, True),
                                  sp_generic=(x, y, True)).items():
        A = torch.round(a * S); B = torch.round(b * S)            # integer encodings
        ea = crypten.cryptensor(A / S, src=0, device=dev)
        if pub:
            z = ea * (B / S).to(dev)                                # public float tensor, re-encoded by CrypTen
        else:
            eb = crypten.cryptensor(B / S, src=1, device=dev)
            z = ea * eb
        Zi = z._tensor.reveal().cpu().double()   # exact ring integer (bypass CrypTen float32 decode)
        err = Zi - A * B / S                                        # ulp
        cases[name] = stats(err)
        # matmul-shaped (sum of 256 products, single truncation) for reference
    a = torch.randn(N // 256 * 0 + 800, 256, generator=g, dtype=torch.float64)
    b = torch.randn(256, 250, generator=g, dtype=torch.float64)
    A = torch.round(a * S); B = torch.round(b * S)
    Zi = (crypten.cryptensor(A / S, src=0, device=dev) @ crypten.cryptensor(B / S, src=1, device=dev))._tensor.reveal().cpu().double()
    cases['ss_matmul'] = stats(Zi - A @ B / S)
    # conditional bias by sign of the true product (update-shaped case)
    A = torch.round(u * S); B = torch.round(lc * S)
    Zi = (crypten.cryptensor(A / S, src=0, device=dev) * crypten.cryptensor(B / S, src=1, device=dev))._tensor.reveal().cpu().double()
    e = Zi - A * B / S; p = (A * B) > 0
    cases['ss_update_by_sign'] = dict(pos=stats(e[p]), neg=stats(e[~p]))
    # decode-bug count with CrypTen get_plain_text on the same tensor
    z = crypten.cryptensor(A / S, src=0, device=dev) * 30.0
    Zi = z._tensor.reveal().cpu().double(); Zd = z.get_plain_text().cpu().double()
    cases['decode_bug'] = dict(n=int(Zi.numel()), n_err_ge_half=int(((Zd - Zi / S).abs() > 0.5).sum()))
    return cases

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    crypten.cfg.mpc.provider = sys.argv[2] if len(sys.argv) > 2 else 'TFP'
    res = run(dev)
    out = dict(N=N, f=16, device=dev, provider=crypten.cfg.mpc.provider, cases=res[0])
    json.dump(out, open('/workspace/veiltune/stage2/t1_trunc.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
