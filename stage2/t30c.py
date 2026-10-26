"""Controlled CPU test of local-truncation failure rate: matmul with public A vs elementwise mul with public y.
ratio = observed bad / expected, expected = sum |true pre-truncation value| / 2^64 (value = out * 2^32)."""
import torch, crypten, crypten.mpc as mpc
@mpc.run_multiprocess(world_size=2)
def run():
    crypten.init(); torch.manual_seed(0); res = []
    for K in (64, 256, 1024, 4096):
        for asc in (0.1, 0.4, 1.6):
            x = (torch.randn(1024, K) * 8).double()
            A = ((torch.rand(16, K) * 2 - 1) * asc).double()
            ref = x @ A.t(); exp = 0.0; bad = 0
            for _ in range(20):
                xs = crypten.cryptensor(x, src=0)
                out = (xs @ A.t()).get_plain_text().double()
                bad += int(((out - ref).abs() > 1.0).sum()); exp += float(ref.abs().sum()) * 2**32 / 2**64
            y = A[0].expand_as(x); refm = x * y; expm = 0.0; badm = 0
            for _ in range(2):
                xs = crypten.cryptensor(x, src=0)
                o = (xs * y).get_plain_text().double()
                badm += int(((o - refm).abs() > 1.0).sum()); expm += float(refm.abs().sum()) * 2**32 / 2**64
            res.append((K, asc, bad, round(exp, 3), badm, round(expm, 3)))
    return res
if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    for r in run()[0]: print('T30C K=%d |A|=%.1f matmul bad %d exp %.3f | mul bad %d exp %.3f' % r)
