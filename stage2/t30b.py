import torch, crypten, crypten.mpc as mpc
@mpc.run_multiprocess(world_size=2)
def run():
    crypten.init(); torch.manual_seed(0); res=[]
    for dev in ('cpu','cuda'):
        for scale in (32.0, 128.0):
            x = (torch.randn(2048, 1024) * scale / 4).double().to(dev); x[:, 7] = scale
            A = (torch.rand(4, 1024) * 0.8 - 0.4).double().to(dev)
            xs = crypten.cryptensor(x, src=0, device=dev)
            bad=0
            for _ in range(10):
                out = (xs @ A.t()).get_plain_text().double(); bad += int(((out - x @ A.t()).abs() > 1.0).sum())
            ys = xs * 3.0; badm = 0
            for _ in range(10):
                badm += int(((( xs * x).get_plain_text().double() - x*x).abs() > 1.0).sum())
            res.append((dev, scale, bad, 10*2048*4, badm, 10*x.numel(), float((x*x).abs().mean())))
    return res
if __name__ == '__main__':
    crypten.cfg.mpc.provider='TTP'; print('T30B', run()[0]); print('prec', crypten.cfg.encoder.precision_bits)
