# VeilTune

Code, data splits and raw experiment logs for the anonymous submission
*VeilTune: Secure Fine-Tuning of Private LoRA Adapters on Public Pretrained Language Models*.

VeilTune runs the forward pass, the backward pass and the Adam update of LoRA fine-tuning entirely in
two-party computation (CrypTen-based, 2PC with a trusted dealer, Z_{2^64}, 16 fractional bits) and reveals only the
trained adapter, to the model owner.

## Layout

| Path | Content |
|---|---|
| `gpu_crypten*.py`, `gpu_crypten_bp2.py` | secure training step: token packing, training-aware nonlinear operators, oblivious step skipping |
| `stage2/` | secure training / timing drivers, per-stage profiler (`t40_*`), user-space network emulation (`t37_netem.py`), run scripts |
| `rl_plain.py`, `rl_leak.py` | plaintext training and the adapter extraction attack (AEA) against SAP / PrivTuner / SecP-Tuning |
| `secp_repro.py`, `cma8/`, `zo8/` | forward-only baselines (CMA-ES, SPSA) |
| `data*/` | task splits (SST-2, Yelp Polarity, AG News, RTE, MRPC); `data_raw/` has the download / split scripts |
| `bench/`, `timing*/` | cost measurements (LAN 3 Gbps / 0.8 ms, WAN 200 Mbps / 40 ms) |
| `hparam/`, `ablation*/` | hyperparameter sensitivity and ablations |
| `rl_leak8/` | attack results (three seeds per task); `agg_seed.py` aggregates them |
| `sec_train*/`, `plain_noskip/` | secure and plaintext accuracy runs |
| `baselines/sigma/` | build script and runs of the SIGMA / GPU-MPC baseline |
| `env/` | Python environment (`requirements.txt`, `environment.yml`) and hardware description |

Files named `*.pre_*.py`, `*.v1.py` etc. are earlier versions kept for traceability.

## Not included (regenerable)

* `rl_cache*/` (about 22 GB): cached hidden states of the frozen lower blocks; rebuilt automatically by `rl_plain.build_cache`.
* RoBERTa-large weights: downloaded from HuggingFace (`FacebookAI/roberta-large`), cached under `cache/huggingface`.
* Raw Yelp Polarity parquet: re-download with `data_raw/dl.py`, then `data_raw/prep.py` and `data_raw/prep_owner.py`.
* SIGMA / GPU-MPC source: https://github.com/mpc-msri/GPU-MPC (see `baselines/sigma/build_sigma.sh`).

## Environment

Python 3.12, PyTorch 2.8 (CUDA 12.8), transformers 5.19, CrypTen; experiments ran on RTX PRO 6000 Blackwell GPUs.
Paths in scripts assume the repository is checked out at `/workspace/veiltune`.
