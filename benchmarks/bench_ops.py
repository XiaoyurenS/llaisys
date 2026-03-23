from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import llaisys  # noqa: E402
import torch  # noqa: E402

from common import llaisys_device, summarize, sync_llaisys, sync_torch, torch_device, write_csv  # noqa: E402


def copy_to_llaisys(tensor: torch.Tensor, device_name: str) -> llaisys.Tensor:
    dtype_map = {
        torch.float32: llaisys.DataType.F32,
        torch.float16: llaisys.DataType.F16,
        torch.bfloat16: llaisys.DataType.BF16,
        torch.int64: llaisys.DataType.I64,
    }
    out = llaisys.Tensor(
        tensor.shape,
        dtype=dtype_map[tensor.dtype],
        device=llaisys_device(device_name),
    )
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(
        out.data_ptr(),
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return out


def bench_pair(torch_fn, llaisys_fn, device_name: str, warmup: int, repeat: int) -> tuple[float, float]:
    for _ in range(warmup):
        torch_fn()
        llaisys_fn()
    sync_torch(device_name)
    sync_llaisys(device_name)

    torch_samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        torch_fn()
        sync_torch(device_name)
        end = time.perf_counter()
        torch_samples.append((end - start) * 1000.0)

    llaisys_samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        llaisys_fn()
        sync_llaisys(device_name)
        end = time.perf_counter()
        llaisys_samples.append((end - start) * 1000.0)

    return summarize(torch_samples)["mean"], summarize(llaisys_samples)["mean"]


def torch_rope(out: torch.Tensor, x: torch.Tensor, pos_ids: torch.Tensor, theta: float):
    half = out.shape[-1] // 2
    xa, xb = x[..., :half], x[..., half:]
    positions = pos_ids.to(torch.float32).unsqueeze(1)
    freqs = positions / (theta ** (2 * torch.arange(0, half, dtype=torch.float32, device=x.device) / x.shape[-1]))
    sin = freqs.sin().unsqueeze(1)
    cos = freqs.cos().unsqueeze(1)
    out[..., :half] = xa * cos - xb * sin
    out[..., half:] = xb * cos + xa * sin


def torch_self_attention(attn_val, query, key, value, scale):
    query = query.transpose(-2, -3)
    key = key.transpose(-2, -3)
    value = value.transpose(-2, -3)
    lq, lk = query.size(-2), key.size(-2)
    mask = torch.ones(lq, lk, dtype=torch.bool, device=query.device).tril(diagonal=lk - lq)
    key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
    value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
    attn_weight = query @ key.transpose(-2, -1) * scale
    attn_weight.masked_fill_(~mask, float("-inf"))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_val.copy_((attn_weight @ value).transpose(-2, -3))


def load_model_meta(model_path: str | None) -> dict[str, int]:
    if model_path is None:
        return {"hs": 1536, "nh": 12, "nkvh": 2, "di": 8960}
    with open(Path(model_path) / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "hs": int(cfg["hidden_size"]),
        "nh": int(cfg["num_attention_heads"]),
        "nkvh": int(cfg.get("num_key_value_heads", cfg["num_attention_heads"])),
        "di": int(cfg["intermediate_size"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--dtype", default="bf16", choices=["f32", "f16", "bf16"])
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    meta = load_model_meta(args.model)
    hs = meta["hs"]
    nh = meta["nh"]
    nkvh = meta["nkvh"]
    di = meta["di"]
    dh = hs // nh

    torch_dtype_map = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
    dtype = torch_dtype_map[args.dtype]
    device = torch_device(args.device)

    def rand(shape):
        return torch.rand(shape, dtype=dtype, device=device)

    rows: list[dict[str, object]] = []

    x = rand((args.tokens, hs))
    w_attn = rand((hs, hs))
    b_attn = rand((hs,))
    out_linear = rand((args.tokens, hs))
    out_linear_ = copy_to_llaisys(out_linear, args.device)
    x_ = copy_to_llaisys(x, args.device)
    w_attn_ = copy_to_llaisys(w_attn, args.device)
    b_attn_ = copy_to_llaisys(b_attn, args.device)
    def torch_linear():
        out_linear.copy_(torch.nn.functional.linear(x, w_attn, b_attn))

    torch_ms, llaisys_ms = bench_pair(
        torch_linear,
        lambda: llaisys.Ops.linear(out_linear_, x_, w_attn_, b_attn_),
        args.device,
        args.warmup,
        args.repeat,
    )
    rows.append({
        "operator": "linear_attn",
        "config": f"tokens={args.tokens},hs={hs}",
        "torch_ms_mean": round(torch_ms, 3),
        "llaisys_ms_mean": round(llaisys_ms, 3),
        "gap_vs_torch": round(llaisys_ms / torch_ms, 3),
    })

    rms_w = rand((hs,))
    rms_out = rand((args.tokens, hs))
    rms_out_ = copy_to_llaisys(rms_out, args.device)
    rms_w_ = copy_to_llaisys(rms_w, args.device)

    def torch_rms():
        tmp = x.float().pow(2).mean(dim=-1, keepdim=True).add(1e-5).rsqrt()
        rms_out.copy_((x * tmp.to(dtype)) * rms_w)

    torch_ms, llaisys_ms = bench_pair(
        torch_rms,
        lambda: llaisys.Ops.rms_norm(rms_out_, x_, rms_w_, 1e-5),
        args.device,
        args.warmup,
        args.repeat,
    )
    rows.append({
        "operator": "rms_norm",
        "config": f"tokens={args.tokens},hs={hs}",
        "torch_ms_mean": round(torch_ms, 3),
        "llaisys_ms_mean": round(llaisys_ms, 3),
        "gap_vs_torch": round(llaisys_ms / torch_ms, 3),
    })

    rope_in = rand((args.tokens, nh, dh))
    rope_out = rand((args.tokens, nh, dh))
    pos_ids = torch.arange(args.tokens, dtype=torch.int64, device=device)
    rope_in_ = copy_to_llaisys(rope_in, args.device)
    rope_out_ = copy_to_llaisys(rope_out, args.device)
    pos_ids_ = copy_to_llaisys(pos_ids, args.device)
    torch_ms, llaisys_ms = bench_pair(
        lambda: torch_rope(rope_out, rope_in, pos_ids, 10000.0),
        lambda: llaisys.Ops.rope(rope_out_, rope_in_, pos_ids_, 10000.0),
        args.device,
        args.warmup,
        args.repeat,
    )
    rows.append({
        "operator": "rope",
        "config": f"tokens={args.tokens},nh={nh},dh={dh}",
        "torch_ms_mean": round(torch_ms, 3),
        "llaisys_ms_mean": round(llaisys_ms, 3),
        "gap_vs_torch": round(llaisys_ms / torch_ms, 3),
    })

    q = rand((args.tokens, nh, dh))
    k = rand((args.tokens, nkvh, dh))
    v = rand((args.tokens, nkvh, dh))
    attn = rand((args.tokens, nh, dh))
    q_ = copy_to_llaisys(q, args.device)
    k_ = copy_to_llaisys(k, args.device)
    v_ = copy_to_llaisys(v, args.device)
    attn_ = copy_to_llaisys(attn, args.device)
    scale = 1.0 / (dh ** 0.5)
    torch_ms, llaisys_ms = bench_pair(
        lambda: torch_self_attention(attn, q, k, v, scale),
        lambda: llaisys.Ops.self_attention(attn_, q_, k_, v_, scale),
        args.device,
        args.warmup,
        args.repeat,
    )
    rows.append({
        "operator": "self_attention",
        "config": f"tokens={args.tokens},nh={nh},nkvh={nkvh},dh={dh}",
        "torch_ms_mean": round(torch_ms, 3),
        "llaisys_ms_mean": round(llaisys_ms, 3),
        "gap_vs_torch": round(llaisys_ms / torch_ms, 3),
    })

    gate = rand((args.tokens, di))
    up = rand((args.tokens, di))
    swiglu_out = rand((args.tokens, di))
    gate_ = copy_to_llaisys(gate, args.device)
    up_ = copy_to_llaisys(up, args.device)
    swiglu_out_ = copy_to_llaisys(swiglu_out, args.device)
    torch_ms, llaisys_ms = bench_pair(
        lambda: swiglu_out.copy_(up * (gate / (1 + torch.exp(-gate.float()).to(dtype)))),
        lambda: llaisys.Ops.swiglu(swiglu_out_, gate_, up_),
        args.device,
        args.warmup,
        args.repeat,
    )
    rows.append({
        "operator": "swiglu",
        "config": f"tokens={args.tokens},di={di}",
        "torch_ms_mean": round(torch_ms, 3),
        "llaisys_ms_mean": round(llaisys_ms, 3),
        "gap_vs_torch": round(llaisys_ms / torch_ms, 3),
    })

    for row in rows:
        print(row)
    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
