from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType, llaisysDeviceType_t
from ..libllaisys.qwen2 import LlaisysQwen2Meta, llaisysQwen2Model_t
from ..tensor import Tensor

from pathlib import Path
import json
import ctypes
import numpy as np
import safetensors


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU, use_kv_cache: bool = True):
        self._device = device
        self._model: llaisysQwen2Model_t | None = None
        self._end_token = None
        self._use_kv_cache = use_kv_cache
        print("[LLAISYS] qwen2 python init")

        model_path = Path(model_path)

        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        torch_dtype = str(cfg.get("torch_dtype", "float32")).lower()
        if "bfloat16" in torch_dtype:
            dtype = DataType.BF16
        elif "float16" in torch_dtype:
            dtype = DataType.F16
        else:
            dtype = DataType.F32

        nlayer = int(cfg.get("num_hidden_layers", 0))
        hs = int(cfg.get("hidden_size", 0))
        nh = int(cfg.get("num_attention_heads", 0))
        nkvh = int(cfg.get("num_key_value_heads", nh))
        di = int(cfg.get("intermediate_size", 0))
        maxseq = int(cfg.get("max_position_embeddings", 0))
        voc = int(cfg.get("vocab_size", 0))
        epsilon = float(cfg.get("rms_norm_eps", 1e-6))
        theta = float(cfg.get("rope_theta", 10000.0))
        end_token = int(cfg.get("eos_token_id", 0))
        self._end_token = end_token

        dh = int(hs // nh) if nh else 0

        meta = LlaisysQwen2Meta(
            dtype=dtype,
            nlayer=nlayer,
            hs=hs,
            nh=nh,
            nkvh=nkvh,
            dh=dh,
            di=di,
            maxseq=maxseq,
            voc=voc,
            epsilon=epsilon,
            theta=theta,
            end_token=end_token,
        )
        print(
            "[LLAISYS] qwen2 meta:",
            f"dtype={meta.dtype} nlayer={meta.nlayer} hs={meta.hs} nh={meta.nh} nkvh={meta.nkvh} "
            f"dh={meta.dh} di={meta.di} maxseq={meta.maxseq} voc={meta.voc} "
            f"epsilon={meta.epsilon} theta={meta.theta} end_token={meta.end_token}",
        )
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            meta, llaisysDeviceType_t(device), None, 0
        )
        LIB_LLAISYS.llaisysQwen2ModelSetKVCache(self._model, int(use_kv_cache))

        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        if not weights_ptr:
            raise RuntimeError("Failed to get qwen2 weights handle")
        self._weights_ptr = weights_ptr
        weights = weights_ptr.contents

        self._tensors = []

        def dtype_from_numpy(np_dtype):
            name = str(np_dtype).lower()
            if np_dtype == np.float32 or "float32" in name:
                return DataType.F32
            if np_dtype == np.float16 or "float16" in name:
                return DataType.F16
            if "bfloat16" in name:
                return DataType.BF16
            return DataType.F32

        def make_tensor_from_numpy(arr, dtype_override=None):
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            dtype = dtype_override or dtype_from_numpy(arr.dtype)
            if dtype == DataType.F32 and arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            t = Tensor(arr.shape, dtype=dtype, device=device)
            t.load(ctypes.c_void_p(arr.ctypes.data))
            self._tensors.append(t)
            return t.lib_tensor()

        def set_layer_weight(ptr, layer_idx, value):
            arr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p * nlayer)).contents
            arr[layer_idx] = value

        loaded = 0
        skipped = 0
        top_level_map = {
            "model.embed_tokens.weight": "in_embed",
            "lm_head.weight": "out_embed",
            "model.norm.weight": "out_norm_w",
        }
        layer_map = {
            "input_layernorm.weight": "attn_norm_w",
            "self_attn.q_proj.weight": "attn_q_w",
            "self_attn.q_proj.bias": "attn_q_b",
            "self_attn.k_proj.weight": "attn_k_w",
            "self_attn.k_proj.bias": "attn_k_b",
            "self_attn.v_proj.weight": "attn_v_w",
            "self_attn.v_proj.bias": "attn_v_b",
            "self_attn.o_proj.weight": "attn_o_w",
            "post_attention_layernorm.weight": "mlp_norm_w",
            "mlp.gate_proj.weight": "mlp_gate_w",
            "mlp.up_proj.weight": "mlp_up_w",
            "mlp.down_proj.weight": "mlp_down_w",
        }

        def load_weight_tensor(data_pt, name):
            import torch

            t = data_pt.get_tensor(name).contiguous()
            if t.dtype == torch.bfloat16:
                arr_u16 = t.view(torch.uint16).cpu().numpy()
                return make_tensor_from_numpy(arr_u16, dtype_override=DataType.BF16)
            arr = t.cpu().numpy()
            return make_tensor_from_numpy(arr)

        for file in sorted(model_path.glob("*.safetensors")):
            data_pt = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_pt.keys():
                if name_ in top_level_map:
                    setattr(weights, top_level_map[name_], load_weight_tensor(data_pt, name_))
                    loaded += 1
                    continue

                if name_.startswith("model.layers."):
                    parts = name_.split(".")
                    if len(parts) < 4:
                        skipped += 1
                        continue
                    layer_idx = int(parts[2])
                    if layer_idx < 0 or layer_idx >= nlayer:
                        raise ValueError(f"Invalid layer index in weight name: {name_}")
                    suffix = ".".join(parts[3:])
                    attr = layer_map.get(suffix)
                    if not attr:
                        skipped += 1
                        continue
                    set_layer_weight(getattr(weights, attr), layer_idx, load_weight_tensor(data_pt, name_))
                    loaded += 1
                    continue

                skipped += 1

        print(f"[LLAISYS] qwen2 weights loaded: {loaded}, skipped: {skipped}")


    def __del__(self):
        if self._model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def set_kv_cache(self, enabled: bool):
        if self._model is None:
            raise RuntimeError("Model is not initialized")
        self._use_kv_cache = enabled
        LIB_LLAISYS.llaisysQwen2ModelSetKVCache(self._model, int(enabled))

    def reset_cache(self):
        if self._model is None:
            raise RuntimeError("Model is not initialized")
        LIB_LLAISYS.llaisysQwen2ModelResetCache(self._model)

    def infer_next(self, tokens: Sequence[int]) -> int:
        if self._model is None:
            raise RuntimeError("Model is not initialized")
        arr = (ctypes.c_int64 * len(tokens))(*tokens)
        return int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, len(tokens)))

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        if self._model is None:
            raise RuntimeError("Model is not initialized")
        if max_new_tokens is None:
            max_new_tokens = 1
        tokens = [int(x) for x in inputs]
        capacity = len(tokens) + max_new_tokens
        buf = (ctypes.c_int64 * max(1, capacity))()
        for i, token in enumerate(tokens):
            buf[i] = token

        # generate 走 backend-side 循环，减少 Python/ctypes 在每个 token 上的往返开销。
        total = int(LIB_LLAISYS.llaisysQwen2ModelGenerate(
            self._model,
            buf,
            len(tokens),
            max_new_tokens,
            capacity,
        ))
        if total <= 0:
            raise RuntimeError("Backend generate failed")
        return [int(buf[i]) for i in range(total)]
