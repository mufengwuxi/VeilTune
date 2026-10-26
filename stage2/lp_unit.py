import sys, json, math, torch, crypten, crypten.mpc as mpc
sys.path.insert(0, '/workspace/veiltune')
@mpc.run_multiprocess(world_size=2)
def run():
    import gpu_crypten as G, gpu_crypten_v2 as V, gpu_crypten_lp as LP
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    crypten.cfg.communicator.verbose = True
    c = crypten.communicator.get(); rank = c.get_rank()
    z = torch.cat([torch.linspace(-300, 300, 60001), torch.randn(20000) * 3]).double().cuda()
    x = crypten.cryptensor(z, device='cuda')
    c.reset_communication_stats(); b = LP.ltz_lowprec(x, 16, 32); r = dict(rounds=c.comm_rounds, bpe=c.comm_bytes / z.numel())
    p = b.get_plain_text(); w = (p != (z < 0).double())
    r['wrong'] = int(w.sum()); r['max_abs_wrong'] = float(z[w].abs().max()) if w.any() else 0.0
    c.reset_communication_stats(); b0 = x._ltz(); r['native_bpe'] = c.comm_bytes / z.numel()
    return dict(rank=rank, r=r)
if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    for o in run():
        if o and o['rank'] == 1: print('UNIT', json.dumps(o['r']))
