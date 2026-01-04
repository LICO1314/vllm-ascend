#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from typing import Any, Dict, List, Optional

import torch
import torch_npu
from vllm.attention.backends.abstract import AttentionType

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, COMPRESSED_TENSORS_METHOD,
                               AscendDeviceType, get_ascend_device_type,
                               get_weight_prefetch_method, maybe_trans_nz)


def quant_per_tensor(in_tensor: torch.Tensor,
                     input_scale: torch.Tensor,
                     input_offset: torch.Tensor,
                     function=False):
    return torch_npu.npu_quantize(in_tensor, input_scale, input_offset,
                                  torch.qint8, -1, function)


class AscendW8A8LinearMethod:
    """Linear method for Ascend W8A8.

    Args:
        w_sym: whether the linear weight is symmetrically quantized.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_weight(
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        params_dict["input_scale"] = torch.empty(1, dtype=params_dtype)
        params_dict["input_offset"] = torch.empty(1, dtype=torch.int8)
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        if params_dtype == torch.bfloat16:
            params_dict["deq_scale"] = torch.empty(output_size,
                                                   dtype=torch.float32)
        elif params_dtype == torch.float16:
            params_dict["deq_scale"] = torch.empty(output_size,
                                                   dtype=torch.int64)
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=params_dtype)
        return params_dict

    def get_pergroup_param(self,
                           input_size: int,
                           output_size: int,
                           params_dtype: torch.dtype,
                           layer_type: Optional[str] = None) -> Dict[str, Any]:
        return {}

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        if x.dtype != torch.int8:
            layer_cls_name = layer.__class__.__name__
            weight_prefetch_method = get_weight_prefetch_method()
            # prefetch qkvo_proj.weight preprocess
            if weight_prefetch_method:
                weight_prefetch_method.maybe_prefetch_attn_weight_preprocess(
                    layer_cls_name=layer_cls_name,
                    weight=layer.weight,
                    start_flag=x,
                )
            try:
                quant_comm_config = getattr(layer, "_quant_comm_config")
            except AttributeError:
                quant_comm_config = {}
            comm_fn = quant_comm_config.get("communication_fn")
            enable_flashcomm2_quant_comm = comm_fn is not None and (
                "o_proj" in layer.prefix or "out_proj" in layer.prefix)
            if enable_flashcomm2_quant_comm:
                quant_input_x = x.contiguous().view(
                    -1, layer.aclnn_input_scale_reciprocal.size(0))
                quant_x = torch.ops.vllm.quantize(
                    quant_input_x,
                    layer.aclnn_input_scale,
                    layer.aclnn_input_scale_reciprocal,
                    layer.aclnn_input_offset,
                )
                comm_input = quant_x.view(x.size(0), -1)
                assert comm_fn is not None
                x = comm_fn(comm_input)
            else:
                # quant
                x = torch.ops.vllm.quantize(
                    x,
                    layer.aclnn_input_scale,
                    layer.aclnn_input_scale_reciprocal,
                    layer.aclnn_input_offset,
                )

            # prefetch qkvo_proj.weight postprocess
            if weight_prefetch_method:
                weight_prefetch_method.maybe_prefetch_attn_weight_postprocess(
                    layer_cls_name=layer_cls_name,
                    stop_flag=x,
                )

        quant_bias = layer.quant_bias if tp_rank == 0 else None

        try:
            ascend_quant_method = getattr(layer, "ascend_quant_method")
        except AttributeError:
            ascend_quant_method = ""
        if ascend_quant_method == COMPRESSED_TENSORS_METHOD:
            quant_bias = bias

        if get_ascend_device_type() == AscendDeviceType._310P:
            # On 300I Duo platform, we need transpose again if
            # using nz. This transpose can be skipped in torchair.
            output = torch_npu.npu_quant_matmul(
                x,
                layer.weight.data.transpose(1, 0),
                layer.deq_scale,
                bias=quant_bias,
                output_dtype=layer.params_dtype,
            )
        else:
            output = torch_npu.npu_quant_matmul(
                x,
                layer.weight,
                layer.deq_scale,
                bias=quant_bias,
                output_dtype=layer.params_dtype,
            )
        return output

    def process_weights_after_loading(self, layer):
        expanding_factor = layer.weight.data.shape[1]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False)
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False)
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor),
            requires_grad=False).to(layer.aclnn_input_scale.dtype)
        if get_ascend_device_type() != AscendDeviceType._310P:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = maybe_trans_nz(layer.weight.data)
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
        ascend_quant_method = getattr(layer, "ascend_quant_method", "")
        if ascend_quant_method == COMPRESSED_TENSORS_METHOD:
            deq_scale = layer.input_scale.data * layer.weight_scale.data
            layer.deq_scale = torch.nn.Parameter(deq_scale,
                                                 requires_grad=False)


class AscendW8A8C8KVCacheMethod:
    """C8 KV Cache quantization method for W8A8 models with int8 KV cache."""

    def __init__(self) -> None:
        """Initialize W8A8C8 KV Cache method."""
        self.antiquant_scale_comb = None
        self.key_cache = None
        self.value_cache = None
        from vllm.config import get_current_vllm_config
        vllm_config = get_current_vllm_config()
        self.params_dtype = vllm_config.model_config.dtype

    def create_weights(self, layer) -> None:
        """Create KV cache quantization parameters for Attention layer.
        
        For Qwen2/Qwen3 models with qkv_proj fusion, the KV cache scales in checkpoint
        are named as k_proj.kv_cache_scale and v_proj.kv_cache_scale, which need to be
        mapped to key_antiquant_scale and value_antiquant_scale in the model.
        """
        from vllm.model_executor.utils import set_weight_attrs
        
        scale_dtype = self.params_dtype

        # Create parameters for key and value antiquant scales
        key_scale = torch.empty(layer.num_kv_heads * layer.head_size,
                               dtype=scale_dtype,
                               requires_grad=False)
        value_scale = torch.empty(layer.num_kv_heads * layer.head_size,
                                 dtype=scale_dtype,
                                 requires_grad=False)

        # Register parameters
        key_param = torch.nn.Parameter(key_scale, requires_grad=False)
        value_param = torch.nn.Parameter(value_scale, requires_grad=False)
        
        layer.register_parameter("key_antiquant_scale", key_param)
        layer.register_parameter("value_antiquant_scale", value_param)
        
        # Set custom weight loaders that handle TP sharding and dtype conversion
        def key_weight_loader(param: torch.nn.Parameter, 
                             loaded_weight: torch.Tensor,
                             *args, **kwargs) -> None:
            self._load_kv_cache_scale(param, loaded_weight)
            
        def value_weight_loader(param: torch.nn.Parameter,
                               loaded_weight: torch.Tensor,
                               *args, **kwargs) -> None:
            self._load_kv_cache_scale(param, loaded_weight)
        
        # Mark parameters with custom weight loaders
        # The actual weight name mapping (k_proj.kv_cache_scale -> key_antiquant_scale)
        # should be handled by the model's load_weights method or vllm's standard remapping
        set_weight_attrs(key_param, {"weight_loader": key_weight_loader})
        set_weight_attrs(value_param, {"weight_loader": value_weight_loader})

    def _load_kv_cache_scale(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        """Load KV cache scale with TP sharding support.
        
        This method handles the mapping from checkpoint weights:
        - k_proj.kv_cache_scale -> key_antiquant_scale
        - v_proj.kv_cache_scale -> value_antiquant_scale
        """
        from vllm.distributed import (get_tensor_model_parallel_rank,
                                      get_tensor_model_parallel_world_size)
        
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        # Apply tensor parallel sharding if needed
        if tp_size > 1:
            shard_size = loaded_weight.shape[0] // tp_size
            start_idx = tp_rank * shard_size
            end_idx = (tp_rank + 1) * shard_size
            loaded_weight = loaded_weight[start_idx:end_idx]

        # Convert dtype if needed
        if loaded_weight.dtype != param.dtype:
            loaded_weight = loaded_weight.to(param.dtype)

        param.data.copy_(loaded_weight)

    def process_weights_after_loading(self, layer):
        """Process weights after loading from checkpoint."""
        self.antiquant_scale_comb = torch.cat(
            (layer.key_antiquant_scale.data.unsqueeze(0),
             layer.value_antiquant_scale.data.unsqueeze(0)),
            dim=0).contiguous()

    def anti_quant_int8(self, key_cache, value_cache,
                        layer) -> List[torch.Tensor]:
        """Dequantize int8 KV cache to float."""
        dst_type = self.params_dtype
        assert key_cache.dtype == torch.int8
        assert value_cache.dtype == torch.int8
        assert dst_type != torch.int8

        key_cache_anti_quant = torch_npu.npu_anti_quant(
            x=key_cache,
            scale=layer.key_antiquant_scale.data.view(-1),
            dst_dtype=dst_type)
        value_cache_anti_quant = torch_npu.npu_anti_quant(
            x=value_cache,
            scale=layer.value_antiquant_scale.data.view(-1),
            dst_dtype=dst_type)

        return [key_cache_anti_quant, value_cache_anti_quant]

    def apply(self, layer, query, key, value, kv_cache, attn_metadata,
              attn_type, scale, output) -> torch.Tensor:
        """Apply C8 KV cache quantization during forward pass."""
        num_tokens = query.shape[0]

        if attn_metadata is None:
            return output.view(num_tokens, layer.num_heads * layer.head_size)

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention "
                "are not implemented for AscendW8A8C8KVCacheMethod")

        # Quantize key and value
        quant_key = quant_per_tensor(
            key.view(-1, layer.num_kv_heads * layer.head_size),
            layer.key_antiquant_scale.data.view(-1), None, True)
        quant_value = quant_per_tensor(
            value.view(-1, layer.num_kv_heads * layer.head_size),
            layer.value_antiquant_scale.data.view(-1), None, True)

        query = query.view(-1, layer.num_heads, layer.head_size)
        key = key.view(-1, layer.num_kv_heads, layer.head_size)
        value = value.view(-1, layer.num_kv_heads, layer.head_size)
        value = value.contiguous()

        # Store quantized KV to cache
        if kv_cache[0].numel() > 0:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping

            block_size = key_cache.shape[1]
            slots_indices = slots.reshape(-1, 1)
            block_indices = slots_indices // block_size
            slots_indices = slots_indices % block_size
            indices = torch.cat((block_indices, slots_indices), dim=1)

            torch_npu.npu_scatter_nd_update_(key_cache, indices, quant_key)
            torch_npu.npu_scatter_nd_update_(value_cache, indices, quant_value)

            self.key_cache = key_cache
            self.value_cache = value_cache

        # Handle different attention states
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            # Direct flash attention without cache
            assert attn_metadata.attn_mask is not None
            mask = attn_metadata.attn_mask
            torch_npu._npu_flash_attention(query=query,
                                           key=key,
                                           value=value,
                                           mask=mask,
                                           seq_len=attn_metadata.seq_lens,
                                           scale_value=scale,
                                           num_heads=layer.num_heads,
                                           num_kv_heads=layer.num_kv_heads,
                                           out=output.reshape(query.shape))

        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            # Explicit dequantization for prefix cache
            assert attn_metadata is not None
            assert attn_metadata.attn_mask is not None

            compress_mask = attn_metadata.attn_mask
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _ = self.key_cache.shape  # type: ignore

            key_from_cache = self.key_cache.view(num_block, block_size, -1)
            value_from_cache = self.value_cache.view(num_block, block_size, -1)

            if key_from_cache.dtype == torch.int8:
                key_cache_anti_quant, value_cache_anti_quant = self.anti_quant_int8(
                    key_from_cache, value_from_cache, layer)
            else:
                key_cache_anti_quant = key_from_cache
                value_cache_anti_quant = value_from_cache

            max_seq_len = max(attn_metadata.seq_lens_list) if hasattr(
                attn_metadata, 'seq_lens_list'
            ) and attn_metadata.seq_lens_list is not None else 0

            # Use optimized path for short sequences with block_size=128
            if block_size == 128 and max_seq_len <= 2048:
                key_dq = key_cache_anti_quant.view(  # type: ignore
                    num_block, block_size, -1)
                value_dq = value_cache_anti_quant.view(  # type: ignore
                    num_block, block_size, -1)

                output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key_dq,
                    value=value_dq,
                    atten_mask=compress_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                    num_key_value_heads=layer.num_kv_heads,
                    num_heads=layer.num_heads,
                    scale=scale,
                    sparse_mode=3,
                )
            else:
                # Use general path for other cases
                output = torch_npu._npu_flash_attention_qlens(
                    query=query,
                    key_cache=key_cache_anti_quant,
                    value_cache=value_cache_anti_quant,
                    block_table=block_table,
                    mask=compress_mask,
                    seq_len=attn_metadata.seq_lens,
                    context_lens=attn_metadata.seq_lens,
                    num_kv_heads=layer.num_kv_heads,
                    num_heads=layer.num_heads,
                    scale_value=scale,
                    out=output)

        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            # Fused dequantization in npu_incre_flash_attention
            seq_lens = attn_metadata.seq_lens

            block_size = key_cache.shape[1]
            query = query.view(num_tokens, 1,
                               layer.num_heads * layer.head_size).contiguous()

            key_cache_flatten = key_cache
            value_cache_flatten = value_cache

            output = torch_npu.npu_incre_flash_attention(
                query,
                key_cache_flatten,
                value_cache_flatten,
                num_key_value_heads=layer.num_kv_heads,
                num_heads=layer.num_heads,
                actual_seq_lengths=seq_lens,
                scale_value=scale,
                input_layout='BSH',
                block_size=block_size,
                block_table=attn_metadata.block_tables,
                antiquant_scale=self.antiquant_scale_comb,
            )

        elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            # Explicit dequantization for long context
            if layer.head_size == 192:
                raise NotImplementedError(
                    "KV cache int8 quantization is not implemented for head_size == 192"
                )

            assert attn_metadata is not None
            assert attn_metadata.attn_mask is not None

            if get_ascend_device_type() == AscendDeviceType._310P:
                attn_metadata.attn_mask = \
                    torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(),
                                              ACL_FORMAT_FRACTAL_NZ)
                attn_metadata.seq_lens = \
                    attn_metadata.seq_lens.to(device=query.device)

            num_block, block_size, _ = self.key_cache.shape  # type: ignore
            key_from_cache = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value_from_cache = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)

            if key_from_cache.dtype == torch.int8:
                key_dq, value_dq = self.anti_quant_int8(key_from_cache, value_from_cache, layer)
            else:
                key_dq = key_from_cache
                value_dq = value_from_cache

            output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key_dq,
                value=value_dq,
                atten_mask=attn_metadata.attn_mask,
                block_table=attn_metadata.block_tables,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                num_key_value_heads=layer.num_kv_heads,
                num_heads=layer.num_heads,
                scale=scale,
                sparse_mode=3,
            )
        else:
            raise NotImplementedError(
                f"KV cache int8 quantization is not implemented for attention state: {attn_metadata.attn_state}"
            )

        return output
