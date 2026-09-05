# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch NekoMind model."""
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub.dataclasses import strict
from torch import nn

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
    from fla.utils import tensor_cache
except ImportError as error:
    raise ImportError("Please run `pip install -U fla-core`") from error

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaRMSNorm,
)
from ..mixtral.modeling_mixtral import (
    MixtralExperts,
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
    load_balancing_loss_func,
)
from ..gemma.modeling_gemma import GemmaMLP


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nekocyrene/NekoMind1.5-Base")
@strict
class NekoMindMoeConfig(PreTrainedConfig):
    r"""
    decoder_sparse_step (`int`, *optional*, defaults to 1):
        The frequency of the MoE layer.
    linear_attn_config (`dict`, *optional*):
        K3 KDA configuration. `kda_layers` and `full_attn_layers` use **1-based** layer numbers.
        KDA layers require `short_conv_kernel_size`, `head_dim`, and `num_heads`.
        Optional `use_full_rank_gate` defaults to `False`; `gate_lower_bound` defaults to `None`.
        Layers selected by `kda_layers` use KDA; all other layers use MLA. When unset, all layers use MLA.
    q_lora_rank (`int`, *optional*):
        Query compression rank. When unset, use a direct query projection, as in K3.
    kv_lora_rank (`int`, *optional*):
        MLA KV compression rank; must be supplied for MLA layers.
    qk_nope_head_dim (`int`, *optional*):
        Per-head Q/K dimension; must be supplied for MLA layers.
    qk_rope_head_dim (`int`, *optional*):
        K3's additional Q/shared-K dimension, without rotary encoding. Must be supplied (may be 0).
    v_head_dim (`int`, *optional*):
        Per-head value dimension; must be supplied for MLA layers.
    mla_use_nope (`bool`, *optional*, defaults to `True`):
        Use K3's NoPE attention. MLA asserts that this is enabled.
    mla_use_output_gate (`bool`, *optional*, defaults to `False`):
        Apply K3's sigmoid output gate before the MLA output projection.
    mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):
        Indicate which layers use NekoMindMoeMLP rather than NekoMindMoeSparseMoeBlock
        The list contains layer index, from 0 to num_layers-1 if we have num_layers layers
        If `mlp_only_layers` is empty, `decoder_sparse_step` is used to determine the sparsity.

    ```python
    >>> from transformers import NekoMindMoeModel, NekoMindMoeConfig

    >>> # Initializing a NekoMindMoE style configuration
    >>> configuration = NekoMindMoeConfig()

    >>> # Initializing a model from the NekoMind1.5-Base" style configuration
    >>> model = NekoMindMoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nekomind_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_experts": "num_local_experts",
    }

    # Default tensor parallel plan for base model `NekoMindMoe`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_a_proj": "replicated_with_grad_allreduce",
        "layers.*.self_attn.q_a_layernorm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "replicated_with_grad_allreduce",
        "layers.*.self_attn.kv_a_layernorm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.g_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    mla_use_nope: bool = True
    mla_use_output_gate: bool = False
    linear_attn_config: dict | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attention_dropout: float | int = 0.0
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 768
    shared_expert_intermediate_size: int = 768
    num_experts_per_tok: int = 8
    num_experts: int = 128
    norm_topk_prob: bool = False
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: list[int] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.mlp_only_layers = [] if self.mlp_only_layers is None else self.mlp_only_layers
        if self.linear_attn_config is not None:
            assert self.linear_attn_config["kda_layers"] is not None
            assert self.linear_attn_config["full_attn_layers"] is not None
        super().__post_init__(**kwargs)

    @property
    def is_mla(self):
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    @property
    def is_linear_attn(self) -> bool:
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config["kda_layers"] is not None
                and len(self.linear_attn_config["kda_layers"]) == 0
            )
        )

    def is_kda_layer(self, layer_idx: int):
        return (
            self.linear_attn_config is not None
            and (layer_idx + 1) in self.linear_attn_config["kda_layers"]
        )


def index_first_axis(x, indices):
    return x[indices]


@tensor_cache
def get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    lens = prepare_lens_from_mask(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = lens.max().item()
    cu_seqlens = prepare_cu_seqlens_from_mask(attention_mask)
    return indices, cu_seqlens, max_seqlen_in_batch


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    out = hidden_states.new_zeros((batch_size * seq_len, *hidden_states.shape[1:]))
    out[indices] = hidden_states
    return out.view(batch_size, seq_len, *hidden_states.shape[1:])


class NekoMindMoeDynamicCache:
    """
    Dynamic cache for NekoMind MLA and KDA layers (ported from K3).
    Inspired by Qwen3-Next
    """
    is_compileable = False

    def __init__(self, config: NekoMindMoeConfig):
        super().__init__()
        self.config = config

        if config.linear_attn_config is not None:
            self.layer_types = []
            for i in range(config.num_hidden_layers):
                if config.is_kda_layer(i):
                    self.layer_types.append("linear_attention")
                else:
                    self.layer_types.append("full_attention")
        else:
            self.layer_types = ["full_attention"] * config.num_hidden_layers

        self.is_sliding = [False] * config.num_hidden_layers
        self._seen_tokens = 0

        self.transformer_layers = [
            i for i in range(config.num_hidden_layers) if self.layer_types[i] == "full_attention"
        ]

        linear_layers = [i for i in range(
            config.num_hidden_layers) if self.layer_types[i] == "linear_attention"]
        self.last_linear_layer = linear_layers[-1] if linear_layers else -1

        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    def __len__(self):
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                    0, beam_idx)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                    0, beam_idx)

            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx][0].device
                beam_idx = beam_idx.to(device)
                q_conv, k_conv, v_conv = self.conv_states[layer_idx]
                self.conv_states[layer_idx] = (
                    q_conv.index_select(0, beam_idx),
                    k_conv.index_select(0, beam_idx),
                    v_conv.index_select(0, beam_idx),
                )
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(
                    0, beam_idx)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if not self.transformer_layers:
            return self._seen_tokens
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, query_length: int | torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        kv_offset = 0
        if isinstance(query_length, torch.Tensor):
            query_length = query_length.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    @property
    def has_previous_state(self):
        """We have a previous state if the last linear (conv) layer was already updated."""
        if self.last_linear_layer == -1:
            return False
        return self.conv_states[self.last_linear_layer] is not None


class NekoMindMoeDeltaAttention(nn.Module):
    def __init__(self, config: NekoMindMoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.mode = "chunk"

        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads

        self.layer_idx = layer_idx

        assert self.mode in [
            'chunk', 'fused_recurrent'], f"Not supported mode `{self.mode}`."

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(
            self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation='silu',
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation='silu',
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation='silu',
        )

        self.A_log = torch.nn.Parameter(torch.log(torch.empty(
            self.num_heads, dtype=torch.float32).uniform_(1, 16)))

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(
            torch.empty(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.use_full_rank_gate = config.linear_attn_config.get("use_full_rank_gate", False)
        self.gate_lower_bound = config.linear_attn_config.get("gate_lower_bound", None)
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation='sigmoid')
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params: NekoMindMoeDynamicCache | None = None,
        **kwargs: Unpack[dict],
    ) -> torch.Tensor:
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask")

            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len] "
                    "(0 = padding). 3D masks are not supported here.",
                )
        use_cache = cache_params is not None
        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if use_cache and q_len == 1 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        cu_seqlens = kwargs.get('cu_seqlens')
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None
        if cache_params is not None:
            if cache_params.conv_states[self.layer_idx] is not None:
                conv_state_q, conv_state_k, conv_state_v = cache_params.conv_states[
                    self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        q_proj_states = self.q_proj(hidden_states)
        k_proj_states = self.k_proj(hidden_states)
        v_proj_states = self.v_proj(hidden_states)
        q, conv_state_q = self.q_conv1d(
            x=q_proj_states,
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            x=k_proj_states,
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            x=v_proj_states,
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)
        beta = self.b_proj(hidden_states).float()

        q, k = map(lambda x: rearrange(
            x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if mode == 'chunk':
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=self.gate_lower_bound is not None,
                lower_bound=self.gate_lower_bound,
                transpose_state_layout=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=self.gate_lower_bound,
                transpose_state_layout=True,
                cu_seqlens=cu_seqlens,
            )
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = recurrent_state
            cache_params.conv_states[self.layer_idx] = (
                conv_state_q, conv_state_k, conv_state_v)

        if self.use_full_rank_gate:
            g = self.g_proj(hidden_states)
        else:
            g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)
        o = self.o_norm(o, g)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand the key/value heads from `num_key_value_heads` to `num_attention_heads`."""
    if n_rep == 1:
        return hidden_states
    return torch.repeat_interleave(hidden_states, dim=1, repeats=n_rep)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]]

    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probs = F.dropout(probs, p=dropout, training=module.training)
    out = torch.einsum("bhqk,bhkd->bhqd", probs, value).transpose(1, 2).contiguous()

    return out, probs


class NekoMindMoeMLAAttention(nn.Module):
    """
    Multi-Latent Attention adapted from deepseek-v3
    """

    def __init__(self, config: NekoMindMoeConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        try:
            self.q_lora_rank = config.q_lora_rank
            self.qk_rope_head_dim = config.qk_rope_head_dim
            self.kv_lora_rank = config.kv_lora_rank
            self.v_head_dim = config.v_head_dim
            self.qk_nope_head_dim = config.qk_nope_head_dim
            self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.use_nope = config.mla_use_nope
            self.scaling = self.q_head_dim ** (-0.5)
        except Exception as e:
            raise ValueError(
                f"NekoMind MLA config is not found or not properly formatted: {e}")

        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(
                self.hidden_size, self.q_lora_rank, bias=False,
            )
            self.q_a_layernorm = NekoMindMoeRMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False,
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = NekoMindMoeRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )
        self.is_causal = True
        assert self.use_nope

        self.use_output_gate = getattr(config, "mla_use_output_gate", False)
        if self.use_output_gate:
            projection_size = self.num_heads * self.v_head_dim
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.rotary_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (batch_size, seq_length, -1,
                     self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is not None:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q_states = self.q_proj(hidden_states)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(
            k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(
            k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx)

        if self.config._attn_implementation == "flash_attention_2" and self.q_head_dim != self.v_head_dim:
            value_states = F.pad(
                value_states, [0, self.q_head_dim - self.v_head_dim])

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if self.config._attn_implementation == "flash_attention_2" and self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(
            batch_size, seq_length, -1).contiguous()
        if self.use_output_gate:
            g = self.g_proj(hidden_states).sigmoid()
            attn_output = attn_output * g
        attn_output = self.o_proj(attn_output)
        return attn_output


class NekoMindMoeMLP(GemmaMLP):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]


class NekoMindMoeExperts(MixtralExperts):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts
        self.intermediate_dim = config.moe_intermediate_size


class NekoMindMoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        routing_weights = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_scores = router_top_value.to(routing_weights.dtype)
        return router_logits, router_scores, router_indices


class NekoMindMoeSparseMoeBlock(nn.Module):
    def __init__(self, config: NekoMindMoeConfig):
        super().__init__()
        self.experts = NekoMindMoeExperts(config)
        self.gate = NekoMindMoeTopKRouter(config)
        self.shared_expert = NekoMindMoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output


class NekoMindMoeRMSNorm(LlamaRMSNorm):
    pass


class NekoMindMoeDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: NekoMindMoeConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.is_linear_attn = config.is_kda_layer(layer_idx)
        if self.is_linear_attn:
            self.self_attn = NekoMindMoeDeltaAttention(config, layer_idx)
        elif config.is_mla:
            self.self_attn = NekoMindMoeMLAAttention(config, layer_idx)
        else:
            raise NotImplementedError
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = NekoMindMoeSparseMoeBlock(config)
        else:
            self.mlp = NekoMindMoeMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = NekoMindMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NekoMindMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # K3's KDA returns a tensor and uses its recurrent/conv cache directly.
        if self.is_linear_attn:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_params=past_key_values,
                **kwargs,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class NekoMindMoePreTrainedModel(MixtralPreTrainedModel):
    _is_stateful = True
    _can_compile_fullgraph = False

    _can_record_outputs = {
        "router_logits": OutputRecorder(NekoMindMoeTopKRouter, index=0),
        "hidden_states": NekoMindMoeDecoderLayer,
        "attentions": NekoMindMoeMLAAttention,
    }


class NekoMindMoeModel(MixtralModel):
    def __init__(self, config: NekoMindMoeConfig):
        NekoMindMoePreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [NekoMindMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = NekoMindMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = NekoMindMoeDynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is not None and not isinstance(past_key_values, NekoMindMoeDynamicCache):
            raise TypeError("past_key_values must be a NekoMindMoeDynamicCache")
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=linear_attn_mask if decoder_layer.is_linear_attn else causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        if past_key_values is not None:
            past_key_values._seen_tokens = past_seen_tokens + inputs_embeds.shape[1]
        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class NekoMindMoeForCausalLM(MixtralForCausalLM):
    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def __init__(self, config):
        super().__init__(config)
        self.model = NekoMindMoeModel(config)
        self.num_experts = config.num_experts

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, NekoMindMoeForCausalLM

        >>> model = NekoMindMoeForCausalLM.from_pretrained("nekocyrene/NekoMind1.5-Base")
        >>> tokenizer = AutoTokenizer.from_pretrained("nekocyrene/NekoMind1.5-Base")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class NekoMindMoeForSequenceClassification(LlamaForSequenceClassification):
    pass


class NekoMindMoeForTokenClassification(LlamaForTokenClassification):
    pass


class NekoMindMoeForQuestionAnswering(LlamaForQuestionAnswering):
    pass


__all__ = [
    "NekoMindMoeConfig",
    "NekoMindMoeForCausalLM",
    "NekoMindMoeForQuestionAnswering",
    "NekoMindMoeModel",
    "NekoMindMoePreTrainedModel",
    "NekoMindMoeForSequenceClassification",
    "NekoMindMoeForTokenClassification",
]
