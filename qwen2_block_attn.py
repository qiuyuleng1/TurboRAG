from typing import Any
from transformers import AutoTokenizer, Qwen2ForCausalLM

import json
import os
import torch
import math
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    repeat_kv,
    rotate_half,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
    Qwen2Config,
    Cache
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding  # Import the rotary embedding class
from typing import Optional, Tuple, Union

def apply_single_rotary_pos_emb(inputs, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    embed = (inputs * cos) + (rotate_half(inputs) * sin)
    return embed

def apply_single_reverse_rotary_pos_emb(inputs, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = -1 * sin[position_ids].unsqueeze(unsqueeze_dim)
    embed = (inputs * cos) + (rotate_half(inputs) * sin)
    return embed

def gen_chunk_position_ids(seen_tokens: int):
    # lengths = [88, 524, 62, 524, 524, 426, 311, 80, 499, 522, 524] for test
    lengths = json.loads(os.environ.get("CHUNK_TOKEN_COUNT_LIST"))
    # print(f"lengths = {lengths}")
    chunk_position_ids = []
    current_sum = 0
    for length in lengths:
        chunk_position_ids.extend(range(length))  # Add positions [0, 1, ..., length-1]
        current_sum += length
        if current_sum >= seen_tokens:
            print("INIT ?")
            break
    # print(f"seen_tokens={seen_tokens}, current_sum={current_sum}, remaining_tokens={seen_tokens - current_sum}")
    if current_sum < seen_tokens:
        remaining_tokens = seen_tokens - current_sum
        chunk_position_ids.extend(range(current_sum, current_sum+remaining_tokens))
    chunk_position_ids = torch.tensor(chunk_position_ids, dtype=torch.long, device="cpu")
    chunk_position_ids = chunk_position_ids.unsqueeze(0)
    # print(f"chunk_position_ids = {chunk_position_ids.tolist()}")
    return chunk_position_ids


class Qwen2BlockAttnSdpaAttention(Qwen2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        print("I am block sdpa attn.")
        # print(f"position_ids = {position_ids}")
        # print(f"past_key_value.seen_tokens = {past_key_value.seen_tokens}")
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # print(f"past_kvcache is None? {past_key_value is None}")
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Apply rotary embeddings to query_states and key_states
        query_states = apply_single_rotary_pos_emb(query_states, cos, sin, position_ids)
        key_states = apply_single_rotary_pos_emb(key_states, cos, sin, 
                                                 position_ids)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models, DynamicCache
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if os.environ.get("USE_CHUNK_CACHE") == "reordered_positions":
            # Set RoPE to zero
            chunk_position_ids = gen_chunk_position_ids(past_key_value.seen_tokens)
            key_states = apply_single_reverse_rotary_pos_emb(key_states, cos, sin, chunk_position_ids)

            # RoPE with reordered positions
            full_position_ids = torch.arange(
                    0, past_key_value.seen_tokens, dtype=torch.long, device=query_states.device
                )
            full_position_ids = full_position_ids.unsqueeze(0)
            key_states = apply_single_rotary_pos_emb(key_states, cos, sin, full_position_ids)
        elif os.environ.get("USE_CHUNK_CACHE") == "composite_positions":
            pass
        elif os.environ.get("USE_CHUNK_CACHE") == "false":
            pass
        elif os.environ.get("USE_CHUNK_CACHE") == "test_reverse_RoPE":
            full_position_ids = torch.arange(
                    0, past_key_value.seen_tokens, dtype=torch.long, device=query_states.device
                )
            full_position_ids = full_position_ids.unsqueeze(0)
            key_states1 = apply_single_reverse_rotary_pos_emb(key_states, cos, sin, full_position_ids)
            key_states2 = apply_single_rotary_pos_emb(key_states1, cos, sin, full_position_ids)
            print(f"type k cache = {key_states1[0][0].dtype}")
            if torch.allclose(key_states, key_states2, atol=1e-6):
                print("key_states and key_states2 are approximately equal.")
                pass
            else:
                diff = torch.abs(key_states - key_states2)
                max_diff = torch.max(diff)
                print(f"Maximum difference: {max_diff}")
                print("key_states and key_states2 are NOT approximately equal.")
            key_states = key_states2
        

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

class Qwen2BlockAttnDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # self.self_attn = Qwen2ModifiedAttention(config, layer_idx)
        self.self_attn = Qwen2BlockAttnSdpaAttention(config, layer_idx)

class Qwen2BlockAttnModel(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2BlockAttnDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

class Qwen2BlockAttnForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.model = Qwen2BlockAttnModel(config)