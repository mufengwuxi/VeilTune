"""Round 5 workstream A: float four-row comparison on the small backbone.

Model google/bert_uncased_L-4_H-256_A-4 (L4-H256, 4 heads x 64), SST-2 split
from data/*.json, prompt-based MLM head + label word, sequence length 64,
all dropout disabled (eval mode + zero dropout config), FP32.

Rows (each with its own learning-rate grid, 3 seeds, identical batch streams):
  full    first-order full fine-tune (all params, AdamW)      -> ceiling
  lora    first-order LoRA rank {4,8,16} on Q/V (AdamW)       -> rank capacity
  zo      ZO-LoRA (2 forwards/step, FP64 master, eps=1e-3)    -> ZO cost

Metric per run: dev (tune 512) accuracy evaluated every eval_every steps;
forward accounting counts TRAIN sample-forwards only (full/lora: batch/step,
zo: 2*batch/step). The aggregator derives max accuracy and the minimum
train forwards needed to reach target = full-row max - 2 points, then
ratio = min_fwd(zo) / min_fwd(first-order lora) at the selected rank.

[Reconstructed 2026-10 from session records after the original instance was
 released; byte-level fidelity to the original is not guaranteed, but the
 protocol (model/data/LoRA/seeds/streams) is identical.]
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = 'google/bert_uncased_L-4_H-256_A-4'
LENGTH = 64


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha=16):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def encode(rows, tok):
    suffix = tok.encode('it was [MASK].', add_special_tokens=False)
    assert len(suffix) == 4 and tok.mask_token_id in suffix
    result = []
    for row in rows:
        text = tok.encode(row['sentence'], add_special_tokens=False)
        ids = ([tok.cls_token_id] + text[:LENGTH - len(suffix) - 2] + suffix
               + [tok.sep_token_id])
        assert ids.count(tok.mask_token_id) == 1
        n = len(ids)
        result.append((ids + [tok.pad_token_id] * (LENGTH - n),
                       [1] * n + [0] * (LENGTH - n), ids.index(tok.mask_token_id), row['label']))
    return [torch.tensor([row[i] for row in result], dtype=torch.long) for i in range(4)]


class MiniExp:
    def __init__(self, args):
        self.args = args
        torch.set_num_threads(8)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation='eager').cuda().eval()
        assert self.model.config.hidden_size == 256 and self.model.config.num_hidden_layers == 4
        self.data = {name: encode(json.loads((ROOT / f'data/{name}.json').read_text()), self.tok)
                     for name in ['train', 'tune']}
        self.label_ids = torch.tensor(self.tok.convert_tokens_to_ids(['terrible', 'great']),
                                      device='cuda')
        assert (self.label_ids != self.tok.unk_token_id).all()
        self.batch_rng = torch.Generator().manual_seed(args.seed + 1000)
        self.direction_rng = torch.Generator(device='cuda').manual_seed(args.seed + 2000)
        self.out = Path(args.output)
        self.out.mkdir(parents=True, exist_ok=True)
        if args.row == 'full':
            for p in self.model.parameters():
                p.requires_grad_(True)
            self.params = [p for p in self.model.parameters() if p.requires_grad]
            self.opt = torch.optim.AdamW(self.params, lr=args.lr, weight_decay=0.01)
            self.master = None
        else:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.params = []
            for layer in self.model.bert.encoder.layer:
                for name in ['query', 'value']:
                    module = LoRALinear(getattr(layer.attention.self, name), args.rank)
                    setattr(layer.attention.self, name, module)
                    self.params.extend([module.A, module.B])
            self.model.eval()
            if args.row == 'lora':
                self.opt = torch.optim.AdamW(self.params, lr=args.lr)
                self.master = None
            else:  # zo / fg: FP64 master, no optimizer
                self.opt = None
                self.master = [p.detach().double().clone() for p in self.params]
            ids_ = {id(p) for p in self.params}
            self.param_names = [n for n, p in self.model.named_parameters() if id(p) in ids_]
            assert len(self.param_names) == len(self.params)
        (self.out / 'run.json').write_text(json.dumps(dict(args=vars(args), model=MODEL_ID,
            trainable_parameters=sum(p.numel() for p in self.params))))

    def batch(self):
        return torch.randperm(len(self.data['train'][0]), generator=self.batch_rng)[:self.args.batch]

    def loss_on(self, indices):
        ids, mask, positions, y = (t[indices].cuda() for t in self.data['train'])
        logits = self.model(input_ids=ids, attention_mask=mask).logits
        selected = logits[torch.arange(len(ids), device='cuda'), positions]
        return F.cross_entropy(selected.index_select(-1, self.label_ids), y)

    def loss_and_dir(self, indices, tangents):
        """Forward gradient: returns (L(theta), dL/dtheta . u) in ONE forward pass."""
        from torch.func import functional_call, jvp
        ids, mask, positions, y = (t[indices].cuda() for t in self.data['train'])
        names = self.param_names
        def f(*ps):
            logits = functional_call(self.model, dict(zip(names, ps)), (),
                                     dict(input_ids=ids, attention_mask=mask)).logits
            sel = logits[torch.arange(len(ids), device='cuda'), positions]
            return F.cross_entropy(sel.index_select(-1, self.label_ids), y)
        primals = tuple(p.detach() for p in self.params)
        return jvp(f, primals, tuple(tangents))

    @torch.no_grad()
    def evaluate(self):
        correct = total = 0
        for ix in torch.arange(len(self.data['tune'][0])).split(self.args.batch):
            ids, mask, positions, y = (t[ix].cuda() for t in self.data['tune'])
            logits = self.model(input_ids=ids, attention_mask=mask).logits
            selected = logits[torch.arange(len(ids), device='cuda'), positions]
            pred = selected.index_select(-1, self.label_ids).argmax(-1)
            correct += int((pred == y).sum())
            total += len(y)
        return correct / total

    @torch.no_grad()
    def set_parameters(self, direction=None, epsilon=0):
        for i, (p, master) in enumerate(zip(self.params, self.master)):
            p.copy_(master if direction is None else master + epsilon * direction[i].double())

    def train(self):
        forwards = 0
        record = lambda step, acc: [
            (self.out / 'eval.jsonl').open('a').write(json.dumps(
                dict(step=step, accuracy=acc, train_forwards=forwards)) + '\n'),
            print(json.dumps(dict(step=step, accuracy=acc, train_forwards=forwards)), flush=True)]
        record(0, self.evaluate())
        for step in range(1, self.args.steps + 1):
            start = time.perf_counter()
            ix = self.batch()
            if self.args.row == 'zo':
                q = self.args.qdirs
                us = [[torch.randn(p.shape, device='cuda', generator=self.direction_rng)
                       for p in self.params] for _ in range(q)]
                with torch.no_grad():
                    updates = [torch.zeros_like(m) for m in self.master]
                    for u in us:
                        self.set_parameters(u, self.args.epsilon)
                        plus = self.loss_on(ix).item()
                        self.set_parameters(u, -self.args.epsilon)
                        minus = self.loss_on(ix).item()
                        c = (plus - minus) / (2 * self.args.epsilon)
                        for i, u_i in enumerate(u):
                            updates[i] += c * u_i.double()
                    self.set_parameters()
                    for i, master in enumerate(self.master):
                        self.master[i] = master - self.args.lr * updates[i] / q
                    self.set_parameters()
                forwards += 2 * q * self.args.batch
            elif self.args.row == 'fg':
                q = self.args.qdirs
                us = [[torch.randn(p.shape, device='cuda', generator=self.direction_rng)
                       for p in self.params] for _ in range(q)]
                updates = [torch.zeros_like(m) for m in self.master]
                self.set_parameters()
                for u in us:
                    with torch.no_grad():
                        _, c = self.loss_and_dir(ix, u)
                    c = c.item()
                    for i, u_i in enumerate(u):
                        updates[i] += c * u_i.double()
                for i, master in enumerate(self.master):
                    self.master[i] = master - self.args.lr * updates[i] / q
                self.set_parameters()
                forwards += q * self.args.batch   # one forward carrying a tangent
            else:
                loss = self.loss_on(ix)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                forwards += self.args.batch
            if step % self.args.eval_every == 0 or step == self.args.steps:
                record(step, self.evaluate())
            if self.master is not None and step in self.args.save_at:
                torch.save([m.cpu() for m in self.master], self.out / f'master_{step}.pt')
            if step % 1000 == 0:
                (self.out / 'progress.txt').write_text(
                    f'{step}/{self.args.steps} {time.perf_counter()-start:.4f}s/step\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--row', choices=['full', 'lora', 'zo', 'fg'], required=True)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--epsilon', type=float, default=1e-3)
    parser.add_argument('--qdirs', type=int, default=1)
    parser.add_argument('--save-at', type=int, nargs='*', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    MiniExp(args).train()


if __name__ == '__main__':
    main()
