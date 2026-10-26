"""Micro-test: failure rate of secret x secret matmul vs magnitude of x (2PC, TTP), mimicking inner = x @ A^T."""
import sys, torch, crypten, crypten.mpc as mpc
@mpc.run_multiprocess(world_size=2)
def run():
    crypten.init(); torch.manual_seed(0)
    res = []
    for scale in (1.0, 8.0, 32.0, 128.0):
        x = (torch.randn(4096, 1024) * scale / 4).double().cuda(); x[:, 7] = scale   # one outlier column
        A = (torch.rand(4, 1024) * 0.8 - 0.4).double().cuda()
        xs = crypten.cryptensor(x, src=0, device='cuda'); As = crypten.cryptensor(A, src=1, device='cuda')
        bad = 0; tot = 0
        for _ in range(5):
            out = (xs @ As.t()).get_plain_text().double(); ref = x @ A.t()
            bad += int(((out - ref).abs() > 1.0).sum()); tot += ref.numel()
        outp = (xs @ A.t()).get_plain_text().double(); badp = int(((outp - x @ A.t()).abs() > 1.0).sum())
        res.append((scale, bad, tot, badp, float((x @ A.t()).abs().mean())))
    return res
if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    r = run(); print('T30', r[0])
