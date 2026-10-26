import sys, torch, crypten, crypten.mpc as mpc
S = 2 ** 16
@mpc.run_multiprocess(world_size=2)
def run(dev, kind):
    crypten.init()
    g = torch.Generator().manual_seed(7)
    a = torch.randn(800, 256, generator=g, dtype=torch.float64); b = torch.randn(256, 250, generator=g, dtype=torch.float64)
    A = torch.round(a * S); B = torch.round(b * S)
    ea = crypten.cryptensor(A / S, src=0, device=dev)
    if kind == 'ss':
        z = ea @ crypten.cryptensor(B / S, src=1, device=dev)
    elif kind == 'sp':
        z = ea @ (B / S).to(dev)
    elif kind == 'ss_elem_big':  # elementwise with product magnitude like matmul output (~16)
        z = ea[:, :250] * crypten.cryptensor((B / S).T[:800] * 16 if False else (B / S).repeat(4, 1)[:800, :250] * 16, src=1, device=dev)
        Bx = torch.round((B / S).repeat(4, 1)[:800, :250] * 16 * S)
        Z = torch.round(z.get_plain_text().cpu().double() * S)
        e = Z - A[:, :250] * Bx / S
        return dict(bad=int((e.abs() > 2).sum()), vals=e[e.abs() > 2][:8].tolist())
    Z = torch.round(z.get_plain_text().cpu().double() * S)
    e = Z - A @ B / S
    bad = e.abs() > 2
    ref = (A @ B / S / S)
    return dict(bad=int(bad.sum()), vals=e[bad][:8].tolist(), refvals=ref[bad][:8].tolist(), mean_ok=float(e[~bad].mean()))
if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    for dev in ['cuda', 'cpu']:
        for kind in ['ss', 'sp', 'ss_elem_big']:
            print(dev, kind, run(dev, kind)[0], flush=True)
