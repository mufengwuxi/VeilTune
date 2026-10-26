"""VeilTune-LP with a C-class softmax head (; new file, framework files untouched).
The original head computes the binary logit difference d = s1 - s0 and sigmoid(d) - y, which is exact for two
classes only. Here the head computes the C label-word logits z = n @ dec + decb, p = softmax(z) with the
VeilTune softmax operator, and the gradient (p - y) with a secret one-hot y [B, C]; backward uses dec^T."""
import inspect, textwrap
import numpy as np, torch
import gpu_crypten as G
import gpu_crypten_bp as BPm
import gpu_crypten_lp as LP

_src = textwrap.dedent(inspect.getsource(BPm.SecureOpsBP.loss_grad))
_rep = [
    ("d = nh @ self.hdd + self.hdb\n    p1 = self._sigmoid(d)",
     "d = nh @ self.hdec + self.hdecb\n    p1 = self.softmax_save(d, torch.ones(d.shape, dtype=torch.float64, device=d.device))"),
    ("gnh = gd.unsqueeze(1) * self.hdd", "gnh = gd @ self.hdecT"),
]
for a, b in _rep:
    assert a in _src, a
    _src = _src.replace(a, b)
_ns = dict(BPm.__dict__)
exec(compile(_src, '<loss_grad_mc>', 'exec'), _ns)


class SecureOpsLPMC(LP.SecureOpsLP):
    loss_grad = _ns['loss_grad']

    def init_head(self, H):
        super().init_head(H)
        dec = np.asarray(H['head_dec'], dtype=np.float64)       # [D, C]
        decb = np.asarray(H['head_decb'], dtype=np.float64)     # [C]
        self.hdec = G.enc_pub(dec).cuda()
        self.hdecT = G.enc_pub(np.ascontiguousarray(dec.T)).cuda()
        self.hdecb = G.enc_pub(decb).cuda()
