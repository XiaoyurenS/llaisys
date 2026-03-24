from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import llaisys  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def maybe_load_cudart():
    try:
        return ctypes.CDLL("libcudart.so")
    except OSError:
        return None


def build_prompt(tokenizer, prompt: str) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer.encode(rendered, add_special_tokens=False)

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    return [int(x) for x in ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia"])
    parser.add_argument("--prompt", type=str, default="Who are you?")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    device = llaisys.DeviceType.CPU if args.device == "cpu" else llaisys.DeviceType.NVIDIA
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = llaisys.models.Qwen2(args.model, device, use_kv_cache=True)
    cudart = maybe_load_cudart() if args.device == "nvidia" else None

    input_ids = build_prompt(tokenizer, args.prompt)

    # 预热时只跑 LLAISYS 路径，避免 profile 中混入 HuggingFace 权重加载或前向开销。
    for _ in range(args.warmup):
        model.reset_cache()
        model.generate(input_ids, max_new_tokens=max(1, min(args.max_steps, 4)))
        llaisys.RuntimeAPI(device).device_synchronize()

    model.reset_cache()
    # 用 cudaProfilerStart/Stop 把 nsys 采样窗口收紧到纯 LLAISYS generate，
    # 这样最终报告不会混入模型加载或 tokenizer/HuggingFace 路径。
    if cudart is not None:
        cudart.cudaProfilerStart()
    start = time.perf_counter()
    out = model.generate(input_ids, max_new_tokens=args.max_steps)
    llaisys.RuntimeAPI(device).device_synchronize()
    end = time.perf_counter()
    if cudart is not None:
        cudart.cudaProfilerStop()

    print("=== Pure LLAISYS Inference ===")
    print(f"device={args.device} prompt_tokens={len(input_ids)} output_tokens={len(out)}")
    print(f"elapsed_s={end - start:.6f}")
    print(tokenizer.decode(out, skip_special_tokens=False))


if __name__ == "__main__":
    main()
