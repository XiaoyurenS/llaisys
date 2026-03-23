from __future__ import annotations

import argparse

from common import build_prompt_ids, load_tokenizer, measure_llaisys_decode, summarize, write_csv

import llaisys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    # parser.add_argument("--prompt-lens", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--prompt-lens", nargs="+", type=int, default=[128])
    # parser.add_argument("--new-tokens", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--new-tokens", nargs="+", type=int, default=[32])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    model = llaisys.models.Qwen2(
        args.model,
        llaisys.DeviceType.CPU if args.device == "cpu" else llaisys.DeviceType.NVIDIA,
        use_kv_cache=True,
    )
    rows: list[dict[str, object]] = []

    for prompt_len in args.prompt_lens:
        prompt_ids = build_prompt_ids(tokenizer, prompt_len)
        for new_tokens in args.new_tokens:
            stats: dict[bool, dict[str, float]] = {}
            for enabled in (False, True):
                model.set_kv_cache(enabled)
                for _ in range(args.warmup):
                    measure_llaisys_decode(model, prompt_ids, new_tokens, args.device)

                ttft_samples: list[float] = []
                decode_samples: list[float] = []
                tail_samples: list[float] = []
                total_decode_samples: list[float] = []

                for _ in range(args.repeat):
                    _, timings = measure_llaisys_decode(model, prompt_ids, new_tokens, args.device)
                    ttft_samples.append(timings[0])
                    decode_tail = timings[1:]
                    decode_samples.append(sum(decode_tail) / max(1, len(decode_tail)))
                    last_chunk = decode_tail[-16:] if len(decode_tail) >= 16 else decode_tail
                    tail_samples.append(sum(last_chunk) / max(1, len(last_chunk)))
                    total_decode_samples.append(sum(decode_tail) / 1000.0)

                row = {
                    "prompt_len": prompt_len,
                    "new_tokens": new_tokens,
                    "cache": "on" if enabled else "off",
                    "ttft_ms_mean": round(summarize(ttft_samples)["mean"], 3),
                    "avg_decode_ms_per_token_mean": round(summarize(decode_samples)["mean"], 3),
                    "last16_decode_ms_per_token_mean": round(summarize(tail_samples)["mean"], 3),
                    "total_decode_s_mean": round(summarize(total_decode_samples)["mean"], 4),
                }
                rows.append(row)
                print(row)
                stats[enabled] = {
                    "decode_ms": summarize(decode_samples)["mean"],
                    "decode_s": summarize(total_decode_samples)["mean"],
                }

            speedup_row = {
                "prompt_len": prompt_len,
                "new_tokens": new_tokens,
                "cache": "speedup",
                "ttft_ms_mean": "",
                "avg_decode_ms_per_token_mean": round(stats[False]["decode_ms"] / stats[True]["decode_ms"], 3),
                "last16_decode_ms_per_token_mean": "",
                "total_decode_s_mean": round(stats[False]["decode_s"] / stats[True]["decode_s"], 3),
            }
            rows.append(speedup_row)
            print(speedup_row)

    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
