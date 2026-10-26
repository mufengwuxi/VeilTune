"""[t40] Per-stage profile of one secure training step: crypten rounds/bytes, the message
pattern used by the network model of t37 (latency units in RTTs, bytes per direction), and wall time, split into
input / fwd:{linear,softmax,gelu,layernorm,loss} / bwd / opt (Adam + skipping), plus dealer (TTP) traffic.
Framework files untouched: methods are wrapped at runtime in each worker process."""
import inspect, json, os, sys, time
from collections import defaultdict
import torch
S = dict(label='idle', phase='fwd', depth=0, t=None, step=None, on=False)
ACC = defaultdict(lambda: dict(rounds=0, bytes=0, lat=0.0, bdir=0, s=0.0))
TTP = dict(n=0, bytes=0, s=0.0)
PER = {}
GROUP = {'gelu_save': 'gelu', 'ln_save': 'layernorm', 'softmax_save': 'softmax', '_sigmoid': 'loss',
         'ln_bwd': 'bwd', 'softmax_bwd': 'bwd'}

def _key():
    return S['label'] if S['label'] in ('input', 'opt', 'idle', 'harness') else S['phase'] + ':' + S['label']

def _switch(new_label=None, new_phase=None):
    if S['on']:
        torch.cuda.synchronize(); now = time.perf_counter(); ACC[_key()]['s'] += now - S['t']; S['t'] = now
    if new_phase: S['phase'] = new_phase
    if new_label: S['label'] = new_label

def start_step(step):
    ACC.clear(); TTP.update(n=0, bytes=0, s=0.0); S.update(step=step, phase='fwd', label='input', on=True)
    torch.cuda.synchronize(); S['t'] = time.perf_counter()

def phase(lbl):
    _switch(lbl)

def end_step(rank, out):
    _switch('idle'); S['on'] = False
    PER[S['step']] = dict(acc={k: dict(v) for k, v in ACC.items()}, ttp=dict(TTP))
    os.makedirs(out, exist_ok=True)
    json.dump(PER, open(f'{out}/prof_rank{rank}.json', 'w'), indent=1)

def _wrap_kernel(cls, name):
    raw = inspect.getattr_static(cls, name); is_static = isinstance(raw, staticmethod)
    f = raw.__func__ if is_static else getattr(cls, name)
    def w(*a, **k):
        outer = S['depth'] == 0
        if outer:
            grp = GROUP[name]
            if grp == 'bwd': _switch('nonlin', 'bwd')
            elif name == 'softmax_save':
                x = a[-2] if len(a) >= 2 else a[0]
                _switch('loss' if (hasattr(x, 'dim') and x.dim() == 2) else 'softmax')
            else: _switch(grp)
        S['depth'] += 1
        try: return f(*a, **k)
        finally:
            S['depth'] -= 1
            if outer: _switch('linear')
    setattr(cls, name, staticmethod(w) if is_static else w)

def install(ops):
    import crypten
    from crypten.communicator.distributed_communicator import DistributedCommunicator as DC
    cls = type(ops)
    for n in GROUP:
        if hasattr(cls, n): _wrap_kernel(cls, n)
    olg = DC._log_communication
    def lg(self, nel):
        r0, b0 = self.comm_rounds, self.comm_bytes; olg(self, nel)
        if S['on']:
            a = ACC[_key()]; a['rounds'] += self.comm_rounds - r0; a['bytes'] += self.comm_bytes - b0
    DC._log_communication = lg
    def nel(x):
        if hasattr(x, 'nelement'): return x.nelement()
        if isinstance(x, (list, tuple)): return sum(nel(y) for y in x)
        return 0
    def wrapc(name):
        orig = getattr(DC, name)
        def w(self, *args, **kw):
            res = orig(self, *args, **kw)
            try:
                if S['on'] and self.get_rank() in (0, 1):
                    tg = (getattr(self, 'ttp_group', None), getattr(self, 'ttp_comm_group', None))
                    if any(x is not None and any(x is g for g in tg if g is not None) for x in list(args) + list(kw.values())): return res
                    if name == 'all_reduce': n = nel(args[0]) * self.BYTES_PER_ELEMENT; lat = 1.0
                    elif name in ('send_obj', 'broadcast_obj'): n = sys.getsizeof(args[0]); lat = 0.5
                    elif name == 'recv_obj': n = sys.getsizeof(res); lat = 0.5
                    elif name == 'scatter': n = (nel(res) if res is not None else 0) * self.BYTES_PER_ELEMENT; lat = 0.5
                    elif name == 'barrier': n = 0; lat = 0.5
                    else: n = nel(args[0]) * self.BYTES_PER_ELEMENT; lat = 0.5
                    a = ACC[_key()]; a['lat'] += lat; a['bdir'] += n
            except Exception as e:
                print('PROF_ERR', name, repr(e), flush=True)
            return res
        setattr(DC, name, w)
    for nm in ['send', 'recv', 'isend', 'irecv', 'scatter', 'reduce', 'all_reduce', 'gather', 'all_gather', 'broadcast',
               'barrier', 'send_obj', 'recv_obj', 'broadcast_obj']:
        wrapc(nm)
    try:
        from crypten.mpc.provider.ttp_provider import TTPClient as _TC
        C = _TC._TTPClient__TTPClient; oreq = C.ttp_request
        def req(self, *a, **k):
            t0 = time.perf_counter(); r = oreq(self, *a, **k); dt = time.perf_counter() - t0
            if S['on']: TTP['n'] += 1; TTP['s'] += dt; TTP['bytes'] += nel(r) * 8
            return r
        C.ttp_request = req
    except Exception as e:
        print('PROF_TTP_ERR', repr(e), flush=True)
    import gpu_crypten as G
    orev = G.reveal
    def rev(x):
        if not S['on']: return orev(x)
        prev = S['label']; _switch('harness')
        try: return orev(x)
        finally: _switch(prev)
    G.reveal = rev
