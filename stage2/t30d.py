import torch, crypten, crypten.mpc as mpc
@mpc.run_multiprocess(world_size=2)
def run():
    crypten.init(); torch.manual_seed(0)
    K=4096; x = (torch.randn(1024, K) * 8).double(); A = ((torch.rand(16, K) * 2 - 1) * 1.6).double()
    ref = x @ A.t(); out=[]
    xe = (x*65536).round().long(); Ae = (A*65536).round().long()
    plain_int = (xe @ Ae.t()).double() / 2**32
    for r in range(3):
        xs = crypten.cryptensor(x, src=0); o = (xs @ A.t()).get_plain_text().double()
        idx = ((o - ref).abs() > 1.0).nonzero().tolist()
        out.append([(i, j, float(o[i,j]), float(ref[i,j]), float(plain_int[i,j])) for i, j in idx[:5]])
    return out, float((plain_int-ref).abs().max())
if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'; o, m = run()[0]
    for r in o: print('T30D', r)
    print('T30D plain int max err', m)
