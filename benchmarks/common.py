from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import llaisys  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def torch_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "nvidia":
        return torch.device("cuda:0")
    raise ValueError(f"Unsupported device: {device_name}")


def llaisys_device(device_name: str) -> llaisys.DeviceType:
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    if device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    raise ValueError(f"Unsupported device: {device_name}")


def sync_torch(device_name: str) -> None:
    if device_name == "nvidia":
        torch.cuda.synchronize()


def sync_llaisys(device_name: str) -> None:
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.device_synchronize()


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    if not ordered:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    p95_idx = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_idx],
    }


def write_csv(rows: list[dict[str, object]], output: str | None) -> None:
    if not output or not rows:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_prompt_ids(tokenizer, prompt_len: int) -> list[int]:
    base = (
        "Please explain transformer inference, KV cache, tensor layout, and "
        "memory bandwidth trade-offs in a concise but technical way. "
    )
    token_ids = tokenizer.encode(base, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("Tokenizer produced empty prompt tokens")
    out: list[int] = []
    while len(out) < prompt_len:
        out.extend(token_ids)
    return out[:prompt_len]


def load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_torch_model(model_path: str, device_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = torch_device(device_name)
    model.to(device)
    return model


def perf_ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


def measure_pytorch_decode(model, input_ids: list[int], new_tokens: int, device_name: str) -> tuple[list[int], list[float]]:
    device = torch_device(device_name)
    inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = list(input_ids)
    timings_ms: list[float] = []

    with torch.no_grad():
        start = time.perf_counter()
        outputs = model(input_ids=inputs, use_cache=True)
        sync_torch(device_name)
        end = time.perf_counter()
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        timings_ms.append(perf_ms(start, end))
        generated.append(int(next_token.item()))

        for _ in range(1, new_tokens):
            start = time.perf_counter()
            outputs = model(
                input_ids=next_token,
                use_cache=True,
                past_key_values=past_key_values,
            )
            sync_torch(device_name)
            end = time.perf_counter()
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
            timings_ms.append(perf_ms(start, end))
            generated.append(int(next_token.item()))

    return generated, timings_ms


def measure_llaisys_decode(model, input_ids: list[int], new_tokens: int, device_name: str) -> tuple[list[int], list[float]]:
    model.reset_cache()
    generated = list(input_ids)
    timings_ms: list[float] = []

    start = time.perf_counter()
    next_token = model.infer_next(generated)
    sync_llaisys(device_name)
    end = time.perf_counter()
    timings_ms.append(perf_ms(start, end))
    generated.append(next_token)

    for _ in range(1, new_tokens):
        start = time.perf_counter()
        next_token = model.infer_next(generated)
        sync_llaisys(device_name)
        end = time.perf_counter()
        timings_ms.append(perf_ms(start, end))
        generated.append(next_token)

    return generated, timings_ms
