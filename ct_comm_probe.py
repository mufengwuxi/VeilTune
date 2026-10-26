"""Find the right communicator object holding nonzero comm stats."""
import torch
import crypten
import crypten.mpc as mpc


@mpc.run_multiprocess(world_size=2)
def work():
    crypten.init()
    rank = crypten.communicator.get().get_rank()
    x = torch.randn(64, 64, device='cuda')
    ct = crypten.cryptensor(x, device='cuda')
    y = (ct @ ct).get_plain_text()
    import crypten.communicator as C
    cands = {}
    try:
        cands['get'] = {k: getattr(C.get(), k, None) for k in ('comm_rounds', 'comm_bytes', 'comm_time')}
    except Exception as e:
        cands['get'] = repr(e)[:80]
    for name in dir(C):
        if 'communicator' in name.lower() and not name.startswith('_'):
            obj = getattr(C, name)
            if hasattr(obj, 'comm_bytes'):
                cands[name] = dict(rounds=obj.comm_rounds, bytes=obj.comm_bytes, time=obj.comm_time)
    print(f'RANK{rank}_COMM', cands)


work()
