from __future__ import annotations

import argparse

from common import (
    build_prompt_ids,
    load_tokenizer,
    load_torch_model,
    measure_llaisys_decode,
    measure_pytorch_decode,
    summarize,
    write_csv,
)

import llaisys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--prompt-lens", nargs="+", type=int, default=[32, 128, 512, 1024])
    parser.add_argument("--new-tokens", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    rows: list[dict[str, object]] = []

    torch_model = None
    if not args.skip_pytorch:
        torch_model = load_torch_model(args.model, args.device)

    llaisys_model = llaisys.models.Qwen2(args.model, llaisys.DeviceType.CPU if args.device == "cpu" else llaisys.DeviceType.NVIDIA)

    for prompt_len in args.prompt_lens:
        prompt_ids = build_prompt_ids(tokenizer, prompt_len)
        for new_tokens in args.new_tokens:
            systems = []
            if torch_model is not None:
                systems.append(("PyTorch", lambda: measure_pytorch_decode(torch_model, prompt_ids, new_tokens, args.device)))
            systems.append(("LLAISYS", lambda: measure_llaisys_decode(llaisys_model, prompt_ids, new_tokens, args.device)))

            for name, fn in systems:
                for _ in range(args.warmup):
                    fn()

                ttft_samples: list[float] = []
                decode_samples: list[float] = []
                e2e_samples: list[float] = []
                throughput_samples: list[float] = []

                for _ in range(args.repeat):
                    _, timings = fn()
                    ttft = timings[0]
                    decode = sum(timings[1:]) / max(1, len(timings) - 1)
                    e2e = sum(timings)
                    throughput = (new_tokens * 1000.0) / e2e
                    ttft_samples.append(ttft)
                    decode_samples.append(decode)
                    e2e_samples.append(e2e / 1000.0)
                    throughput_samples.append(throughput)

                ttft_stat = summarize(ttft_samples)
                decode_stat = summarize(decode_samples)
                e2e_stat = summarize(e2e_samples)
                throughput_stat = summarize(throughput_samples)

                row = {
                    "system": name,
                    "device": args.device,
                    "prompt_len": prompt_len,
                    "new_tokens": new_tokens,
                    "ttft_ms_mean": round(ttft_stat["mean"], 3),
                    "ttft_ms_median": round(ttft_stat["median"], 3),
                    "decode_ms_per_token_mean": round(decode_stat["mean"], 3),
                    "decode_ms_per_token_p95": round(decode_stat["p95"], 3),
                    "e2e_s_mean": round(e2e_stat["mean"], 4),
                    "throughput_tok_s_mean": round(throughput_stat["mean"], 3),
                }
                rows.append(row)
                print(row)

    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
