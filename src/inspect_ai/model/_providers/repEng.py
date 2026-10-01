import json
import os
from pathlib import Path
from typing import Any

import torch
from repeng import ControlVector, ControlModel
from transformers import BitsAndBytesConfig
from .hf import HuggingFaceAPI
from .._generate_config import GenerateConfig


class RepEngAPI(HuggingFaceAPI):
    def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            api_key: str | None = None,
            config: GenerateConfig = GenerateConfig(),
            **model_args: Any,
    ):

        # 1. Extract our specific RepEng arguments from the dict
        vector_path = model_args.pop("vector_path", None)
        coefficient = model_args.pop("coefficient", 1.0)

        # ---- Handling BitsAndBytes Quantization Config ----
        load_in_4bit = model_args.pop("load_in_4bit", False)
        load_in_8bit = model_args.pop("load_in_8bit", False)
        bnb_4bit_compute_dtype = model_args.pop("bnb_4bit_compute_dtype", "bfloat16")

        if load_in_4bit or load_in_8bit:
            compute_dtype = (
                torch.bfloat16
                if str(bnb_4bit_compute_dtype).lower() in ["bfloat16", "bf16"]
                else torch.float16
            )

            if load_in_4bit:
                model_args["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            elif load_in_8bit:
                model_args["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

        # 2. Call the super init with the standard parameters
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
            **model_args
        )

        # 3. Apply the steering logic if a vector is provided
        if vector_path is not None and str(vector_path).lower() != "none":
            print(f"DEBUG [RepEng]: Loading ControlVector from {vector_path}")

            # Initial loading onto CPU
            loaded_asset = torch.load(vector_path, map_location="cpu")

            # Detect active model precision
            target_dtype = getattr(self.model, "dtype", torch.bfloat16)

            # FIX 1: If the model is FP8, activation precision must remain bfloat16
            if str(target_dtype).startswith("torch.float8"):
                target_dtype = torch.bfloat16

            # Helper function to identify the exact device for each layer (Multi-GPU Safety)
            def get_layer_device(layer_idx: int) -> torch.device | str:
                try:
                    # Isolate the exact device of the specific sharded layer (Llama / Qwen)
                    return self.model.model.layers[layer_idx].self_attn.q_proj.weight.device
                except (AttributeError, IndexError):
                    # Dynamic fallback
                    dev = getattr(self, "device", "cuda:0")
                    if isinstance(dev, str) and not dev.startswith("cuda") and dev != "cpu":
                        return "cuda:0" if torch.cuda.is_available() else "cpu"
                    return dev

            # Automatically detect model architecture
            if "llama" in model_name.lower():
                detected_arch = "llama"
            elif "qwen" in model_name.lower():
                detected_arch = "qwen"
            elif "gemma" in model_name.lower():
                detected_arch = "gemma"
            else:
                detected_arch = "unknown"

            if isinstance(loaded_asset, dict) and "layer_vectors" in loaded_asset and "layers_involved" in loaded_asset:
                print("DEBUG [RepEng]: Metadata found. Extracting strict layer tracking boundaries...")
                layers = loaded_asset["layers_involved"]
                raw_layer_map = loaded_asset["layer_vectors"]

                model_type_str = loaded_asset.get("model_architecture", detected_arch).lower()
                if "/" in model_type_str:
                    model_type_str = model_type_str.split("/")[-1]
                if "llama" in model_type_str:
                    model_type_str = "llama"
                elif "qwen" in model_type_str:
                    model_type_str = "qwen"
                elif "gemma" in model_type_str:
                    model_type_str = "gemma"
            else:
                print("WARN [RepEng]: Legacy format or missing metadata. Falling back to key discovery.")
                if hasattr(loaded_asset, "keys"):
                    raw_layer_map = {k: loaded_asset[k] for k in loaded_asset.keys()}
                else:
                    raw_layer_map = loaded_asset

                layers = sorted([int(k) for k in raw_layer_map.keys()])
                model_type_str = detected_arch

            # Strict mapping layer -> specific GPU + dtype
            processed_map = {
                int(layer): raw_layer_map[layer].to(
                    device=get_layer_device(int(layer)),
                    dtype=target_dtype
                )
                for layer in layers
            }

            self.vector = ControlVector(model_type_str, directions=processed_map)

            print(f"DEBUG [RepEng]: Hooking exactly into involved layers: {layers}")

            # Wrap model with ControlModel
            self.model = ControlModel(self.model, layers)

            if float(coefficient) != 0:
                self.model.set_control(self.vector, float(coefficient))

                print(f"DEBUG [RepEng]: Steering successfully running (Coeff: {coefficient}) across physical devices.")
