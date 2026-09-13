<!--Copyright 2026 the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-09-13.*

# NekoMindMoE2

## Overview

**NekoMindMoE2** is a sparse Mixture-of-Experts (MoE) language model that combines two attention designs in a single
hybrid stack:

- **Multi-head Latent Attention (MLA)** from [DeepSeek-V2/V3](./deepseek_v3) for the full-attention layers, run in
  **NoPE** mode (`mla_use_nope=True`) so no rotary embedding is applied and the `qk_rope_head_dim` slice of the keys is
  kept but left un-rotated. An optional sigmoid output gate (`mla_use_output_gate`) can further modulate the attention
  output before the output projection.
- **Kimi Delta Attention (KDA)**, a refinement of [Gated DeltaNet](https://huggingface.co/papers/2412.06464) where the
  recurrent state decays **per channel** instead of per token, for the linear-attention layers.

The feed-forward blocks are DeepSeek-style MoE with a shared expert and a per-layer dense/sparse schedule. The exact
interleaving of full-attention, linear-attention, dense and sparse layers is read from the checkpoint: `layer_types`
and `mlp_layer_types` are derived from `linear_attn_config` when it is present, otherwise sensible defaults are used.

This model was contributed by [nekocyrene](https://huggingface.co/nekocyrene).

## Usage examples

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "nekocyrene/NekoMind1.5-Base"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [{"role": "user", "content": "Tell me about the french revolution."}]
model_inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=128)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]

print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

The KDA layers run on a pure PyTorch implementation by default. Installing
[`kernels`](https://github.com/huggingface/kernels) (`pip install -U kernels`) and passing `use_kernels=True`
in `from_pretrained` makes them dispatch to custom kernels instead, which is considerably faster for long sequences.

## NekoMindMoe2Config

[[autodoc]] NekoMindMoe2Config

## NekoMindMoe2Model

[[autodoc]] NekoMindMoe2Model
    - forward

## NekoMindMoe2ForCausalLM

[[autodoc]] NekoMindMoe2ForCausalLM
    - forward
