"""CrypTen GPU smoke: 2 parties, one process group, cuda; f=16/64-bit check,
matmul / comparison / division / TTP availability."""
import torch
import crypten

@crypten.mpc.run_multiprocess(world_size=2)
def smoke():
    crypten.init()
    torch.manual_seed(0)
    x = torch.randn(8, 8, device='cuda').abs() + 0.5
    y = torch.randn(8, 8, device='cuda').abs() + 0.5
    ct_x = crypten.cryptensor(x, device='cuda')
    enc = ct_x.encoder
    print('encoder attrs', [a for a in dir(enc) if 'bit' in a.lower()],
          {a: getattr(enc, a) for a in dir(enc) if a.startswith('_') and 'bit' in a})
    z = (ct_x @ ct_x).get_plain_text()
    print('matmul max err', (z - x @ x).abs().max().item())
    b = (ct_x >= 1.0)
    print('compare ok', bool((b.get_plain_text() == (x >= 1.0)).all()))
    d = (ct_x / ct_x).get_plain_text()
    print('div max err', (d - x / x).abs().max().item())
    r = ct_x.reciprocal().get_plain_text()
    print('recip max err', (r - 1 / x).abs().max().item())
    print('comm bytes', crypten.get_communicator().get_communication_stats()
          if hasattr(crypten.get_communicator(), 'get_communication_stats') else 'n/a')
    try:
        from crypten.mpc.provider import TTPProvider  # noqa: F401
        print('TTPProvider available')
    except Exception as e:
        print('TTPProvider missing:', e)

smoke()
