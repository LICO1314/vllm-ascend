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
"""
Qwen2/Qwen3 模型专用的 C8 KV Cache 量化方法

为什么需要这个新文件？（vs w8a8.py 中的 AscendC8KVCacheMethod）
================================================================
对比项                 | w8a8.py                      | qwen2_c8.py (本文件)
-----------------------|------------------------------|--------------------------------
适用模型               | DeepSeek 等                  | Qwen2/Qwen3
架构特点               | 独立的 q/k/v_proj            | fused qkv_proj
参数注册位置           | 标准位置                     | Attention layer 上直接注册
scale dtype            | 硬编码 float16               | 从 config 读取（支持 bfloat16）
PrefillCacheHit 支持   | ❌ NotImplemented            | ✅ 完整实现
ChunkedPrefill 支持    | ❌ NotImplemented            | ✅ 完整实现
反量化方法             | 内嵌在 apply 中              | 独立的 anti_quant_int8 方法
权重加载               | 标准流程                     | 需要 patch_qwen2_kv_cache.py

核心功能：
---------
- 将 key 和 value 从 bfloat16/float16 量化为 int8 (节省 50% 显存)
- 将 int8 的 KV cache 反量化回 bfloat16/float16 进行计算
- 支持 4 种 attention 状态：PrefillNoCache, PrefillCacheHit, DecodeOnly, ChunkedPrefill
"""

from typing import List, Optional

import torch
import torch_npu
from vllm.attention.backends.abstract import AttentionType
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, AscendDeviceType,
                               get_ascend_device_type)


# ============================================================================
# 【参考 w8a8.py】量化算子封装（与 w8a8.py 相同）
# ============================================================================
def quant_per_tensor(in_tensor: torch.Tensor,
                     input_scale: torch.Tensor,
                     input_offset: Optional[torch.Tensor],
                     function=False):
    """
    Per-tensor 量化：将浮点 tensor 量化为 int8
    
    NPU 算子：torch_npu.npu_quantize
    
    参数：
        in_tensor: 输入 tensor
            - dtype: bfloat16 或 float16
            - shape: 任意 shape，这里通常是 [num_tokens, num_kv_heads * head_size]
        input_scale: 量化的 scale 参数
            - dtype: bfloat16 或 float16
            - shape: [num_kv_heads * head_size] (per-tensor，所以只有一个 scale 向量)
        input_offset: 量化的 offset 参数 (C8 量化不使用，设为 None)
        function: 是否使用函数式调用 (设为 True 避免修改原 tensor)
    
    返回：
        torch.Tensor: 量化后的 int8 tensor
            - dtype: torch.qint8
            - shape: 与输入相同
    
    量化公式：
        quantized_value = round(input_value / scale)
        其中 scale 是从权重中加载的 antiquant_scale
    """
    return torch_npu.npu_quantize(
        in_tensor,       # 输入：bfloat16/float16
        input_scale,     # scale: bfloat16/float16
        input_offset,    # offset: None (C8 不使用)
        torch.qint8,     # 输出类型：int8
        -1,              # axis: -1 表示 per-tensor 量化
        function         # 函数式调用，不修改原 tensor
    )  # 返回：int8


# ============================================================================
# 【我们新增的类】Qwen2C8KVCacheMethod
# 参考了 w8a8.py 的 AscendC8KVCacheMethod，但做了重要扩展
# ============================================================================
class Qwen2C8KVCacheMethod(BaseKVCacheMethod):
    """
    Qwen2/Qwen3 模型的 C8 KV Cache 量化方法
    
    C8 = Cache 8-bit，即将 KV cache 量化为 8-bit (int8)
    
    与 AscendC8KVCacheMethod (w8a8.py) 的主要区别：
    -----------------------------------------------------
    1. ✅ 新增：支持 PrefillCacheHit（Prefix Cache）
    2. ✅ 新增：支持 ChunkedPrefill（长上下文分块）
    3. ✅ 新增：独立的 anti_quant_int8 反量化方法
    4. ✅ 改进：dtype 从 config 读取，支持 bfloat16
    5. ✅ 改进：更灵活的算子选择（根据序列长度自适应）
    """

    # ========================================================================
    # 【参考 w8a8.py，有修改】初始化方法
    # ========================================================================
    def __init__(self, quant_config=None, prefix: str = "") -> None:
        """
        初始化量化方法
        
        vs AscendC8KVCacheMethod:
        【原有】初始化 antiquant_scale_comb
        【新增】从 config 读取 dtype（而不是硬编码 float16）
        """
        self.antiquant_scale_comb = None  # 【原有】组合的 scale，用于 DecodeOnly 阶段
        
        # 【新增】从量化配置中获取模型的 dtype（支持 bfloat16）
        if quant_config and hasattr(quant_config, 'quant_description'):
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()
            self.params_dtype = vllm_config.model_config.dtype  # bfloat16 或 float16
        else:
            self.params_dtype = torch.bfloat16  # 默认使用 bfloat16
        
        # vs AscendC8KVCacheMethod: 硬编码为 float16

    # ========================================================================
    # 【参考 w8a8.py，有修改】创建量化参数
    # ========================================================================
    def create_weights(self, layer) -> None:
        """
        为 Attention layer 创建 KV cache 量化所需的参数
        
        vs AscendC8KVCacheMethod:
        【相同】创建 key_antiquant_scale 和 value_antiquant_scale 参数
        【修改】dtype 使用 self.params_dtype（而不是硬编码 float16）
        
        在模型初始化时调用，为每个 Attention layer 注册量化参数
        
        参数：
            layer: Attention layer 对象
                - layer.num_kv_heads: KV heads 的数量（GQA 中可能少于 Q heads）
                - layer.head_size: 每个 head 的维度（例如 128）
        
        创建的参数：
            - key_antiquant_scale: key 的反量化 scale
                shape: [num_kv_heads * head_size]
                dtype: bfloat16 或 float16（从 config 读取）
                用途：将 int8 的 key cache 反量化回浮点数
            
            - value_antiquant_scale: value 的反量化 scale
                shape: [num_kv_heads * head_size]
                dtype: bfloat16 或 float16
                用途：将 int8 的 value cache 反量化回浮点数
        
        注意：
            这些参数会在后续通过 patch_qwen2_kv_cache.py 从权重文件中加载
        """
        param_dict = {}
        scale_dtype = self.params_dtype  # 【修改】从 config 读取，而非硬编码

        # 创建 key_antiquant_scale 参数
        # 示例：num_kv_heads=8, head_size=128 -> shape=[1024]
        param_dict["key_antiquant_scale"] = torch.empty(
            layer.num_kv_heads * layer.head_size,  # 总共需要的 scale 数量
            dtype=scale_dtype,                      # 【修改】bfloat16/float16（非硬编码）
            requires_grad=False                     # 推理时不需要梯度
        )
        
        # 创建 value_antiquant_scale 参数
        param_dict["value_antiquant_scale"] = torch.empty(
            layer.num_kv_heads * layer.head_size,
            dtype=scale_dtype,
            requires_grad=False
        )

        # 【原有】将参数注册到 layer 上
        # 之后可以通过 layer.key_antiquant_scale 访问
        for weight_name, weight_param in param_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            layer.register_parameter(weight_name, param)

    # ========================================================================
    # 【参考 w8a8.py，有修改】权重加载后处理
    # ========================================================================
    def process_weights_after_loading(self, layer):
        """
        权重加载完成后的后处理
        
        vs AscendC8KVCacheMethod:
        【相同】组合 key 和 value scale
        【修改】移除了 .to(torch.float16)，保留原 dtype
        
        在所有权重加载完毕后调用，进行必要的预处理
        
        功能：
            将 key_antiquant_scale 和 value_antiquant_scale 组合成一个 tensor
            这样在 DecodeOnly 阶段可以一次性传给算子，提高效率
        
        生成的 antiquant_scale_comb:
            shape: [2, num_kv_heads * head_size]
            - [0, :]: key_antiquant_scale
            - [1, :]: value_antiquant_scale
        """
        self.antiquant_scale_comb = torch.cat(
            (layer.key_antiquant_scale.data.unsqueeze(0),   # [1, num_kv_heads * head_size]
             layer.value_antiquant_scale.data.unsqueeze(0)), # [1, num_kv_heads * head_size]
            dim=0                                            # 在第 0 维拼接
        ).contiguous()  # [2, num_kv_heads * head_size]
        # 【修改】保留原 dtype（bfloat16/float16），不强制转为 float16

    # ========================================================================
    # 【我们新增的方法】int8 反量化（AscendC8KVCacheMethod 没有这个独立方法）
    # ========================================================================
    def anti_quant_int8(self, key_cache, value_cache,
                        layer) -> List[torch.Tensor]:
        """
        将 int8 的 KV cache 反量化为浮点数
        
        vs AscendC8KVCacheMethod:
        【新增】独立的反量化方法，便于在多个场景复用
        【原版】AscendC8KVCacheMethod 没有这个方法（因为不支持需要反量化的场景）
        
        NPU 算子：torch_npu.npu_anti_quant (反量化算子)
        
        使用场景：
            - PrefillCacheHit: 需要显式反量化
            - ChunkedPrefill: 需要显式反量化
        
        参数：
            key_cache: int8 量化的 key cache
                - dtype: torch.int8
                - shape: [num_blocks, block_size, num_kv_heads * head_size]
            value_cache: int8 量化的 value cache
                - dtype: torch.int8
                - shape: [num_blocks, block_size, num_kv_heads * head_size]
            layer: Attention layer，用于获取 antiquant_scale
        
        返回：
            [key_cache_anti_quant, value_cache_anti_quant]
            - key_cache_anti_quant: 反量化后的 key
                dtype: bfloat16/float16
                shape: 与输入相同
            - value_cache_anti_quant: 反量化后的 value
                dtype: bfloat16/float16
                shape: 与输入相同
        
        反量化公式：
            dequantized_value = quantized_value * scale
        """
        dst_type = self.params_dtype  # bfloat16 或 float16
        
        # 断言检查输入类型
        assert key_cache.dtype == torch.int8, "key_cache 必须是 int8"
        assert value_cache.dtype == torch.int8, "value_cache 必须是 int8"
        assert dst_type != torch.int8, "目标类型不能是 int8"

        # 反量化 key cache
        # 输入：int8 -> 输出：bfloat16/float16
        key_cache_anti_quant = torch_npu.npu_anti_quant(
            x=key_cache,                                    # int8 输入
            scale=layer.key_antiquant_scale.data.view(-1), # scale: bfloat16/float16, shape=[num_kv_heads * head_size]
            dst_dtype=dst_type                             # 输出类型：bfloat16/float16
        )
        
        # 反量化 value cache
        # 输入：int8 -> 输出：bfloat16/float16
        value_cache_anti_quant = torch_npu.npu_anti_quant(
            x=value_cache,                                    # int8 输入
            scale=layer.value_antiquant_scale.data.view(-1), # scale: bfloat16/float16
            dst_dtype=dst_type                               # 输出类型：bfloat16/float16
        )

        return [key_cache_anti_quant, value_cache_anti_quant]

    # ========================================================================
    # 【参考 w8a8.py，大幅扩展】前向推理应用
    # ========================================================================
    def apply(self, layer, query, key, value, kv_cache, attn_metadata,
              attn_type, scale, output) -> torch.Tensor:
        """
        【重构说明】适配新的 attention 架构（不再使用 AscendAttentionState）
        
        新架构特点：
        - 不再使用 attn_state 枚举
        - 通过 has_decode / has_prefill 判断处理路径
        - Chunked prefill 通过 attn_metadata.prefill.chunked_context 判断
        """
        """
        在前向推理中应用 C8 KV cache 量化
        
        vs AscendC8KVCacheMethod:
        【保留】量化和 cache 写入逻辑（完全相同）
        【保留】PrefillNoCache 和 DecodeOnly 分支（基本相同）
        【新增】PrefillCacheHit 分支（原版直接 raise NotImplementedError）
        【新增】ChunkedPrefill 分支（原版直接 raise NotImplementedError）
        
        这是核心方法，在每次 forward 时被调用
        
        主要流程：
        1. 量化当前的 key/value 为 int8
        2. 将 int8 的 key/value 写入 KV cache
        3. 根据 attention state 选择不同的计算路径
        4. 执行 attention 计算
        
        参数：
            layer: Attention layer 对象
            query: Query tensor
                - dtype: bfloat16/float16
                - shape: [num_tokens, num_heads, head_size]
            key: Key tensor（当前新生成的，未量化）
                - dtype: bfloat16/float16
                - shape: [num_tokens, num_kv_heads, head_size]
            value: Value tensor（当前新生成的，未量化）
                - dtype: bfloat16/float16
                - shape: [num_tokens, num_kv_heads, head_size]
            kv_cache: KV cache tuple (key_cache, value_cache)
                - dtype: torch.int8 (已经是量化后的)
                - shape: [num_blocks, block_size, num_kv_heads * head_size]
            attn_metadata: Attention 元数据，包含各种索引和长度信息
            attn_type: Attention 类型（DECODER/ENCODER等）
            scale: Attention 缩放因子，通常是 1/sqrt(head_dim)
            output: 输出 tensor (预分配的内存)
        
        返回：
            torch.Tensor: Attention 输出
                - dtype: bfloat16/float16
                - shape: [num_tokens, num_heads * head_size]
        """
        num_tokens = query.shape[0]  # 当前批次的 token 数量

        # 【原有】如果没有 attention metadata，直接返回（通常是模型初始化阶段）
        if attn_metadata is None:
            return output.view(num_tokens, layer.num_heads * layer.head_size)

        # 【原有】确保 k_scale 和 v_scale 都是 1.0（不使用额外的缩放）
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0

        # 【原有】目前只支持 Decoder attention（单向 attention）
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention "
                "are not implemented for Qwen2C8KVCacheMethod")

        # ====================================================================
        # 【原有代码 - 与 w8a8.py 完全相同】
        # 步骤1: 量化当前的 key 和 value
        # ====================================================================
        # 输入：bfloat16/float16 -> 输出：int8
        
        quant_key = quant_per_tensor(
            key.view(-1, layer.num_kv_heads * layer.head_size),  # [num_tokens, num_kv_heads * head_size], bfloat16/float16
            layer.key_antiquant_scale.data.view(-1),              # scale: [num_kv_heads * head_size], bfloat16/float16
            None, True                                            # offset=None, function=True
        )  # 返回：int8, shape=[num_tokens, num_kv_heads * head_size]
        
        quant_value = quant_per_tensor(
            value.view(-1, layer.num_kv_heads * layer.head_size),  # bfloat16/float16
            layer.value_antiquant_scale.data.view(-1),             # scale: bfloat16/float16
            None, True
        )  # 返回：int8

        # 【原有】重塑 query, key, value 为 3D tensor
        query = query.view(-1, layer.num_heads, layer.head_size)      # [num_tokens, num_heads, head_size]
        key = key.view(-1, layer.num_kv_heads, layer.head_size)       # [num_tokens, num_kv_heads, head_size]
        value = value.view(-1, layer.num_kv_heads, layer.head_size)   # [num_tokens, num_kv_heads, head_size]
        value = value.contiguous()  # 确保内存连续

        # ====================================================================
        # 【原有代码 - 与 w8a8.py 完全相同】
        # 步骤2: 将量化后的 key/value 写入 KV cache
        # ====================================================================
        if kv_cache[0].numel() > 0:  # 如果 KV cache 已分配
            key_cache, value_cache = kv_cache[0], kv_cache[1]  # 都是 int8 类型
            slots = attn_metadata.slot_mapping  # slot 映射：[num_tokens]，表示每个 token 应该写入哪个 slot

            # 计算 block 索引和 slot 索引
            block_size = key_cache.shape[1]  # 每个 block 的大小（例如 128）
            slots_indices = slots.reshape(-1, 1)  # [num_tokens, 1]
            block_indices = slots_indices // block_size  # block 索引
            slots_indices = slots_indices % block_size   # block 内的 slot 索引
            indices = torch.cat((block_indices, slots_indices), dim=1)  # [num_tokens, 2]

            # 使用 scatter 操作将量化后的 key/value 写入对应位置
            # 输入：int8 -> 写入：int8
            torch_npu.npu_scatter_nd_update_(key_cache, indices, quant_key)
            torch_npu.npu_scatter_nd_update_(value_cache, indices, quant_value)

            # 【新增】保存 cache 引用，供后续 PrefillCacheHit 和 ChunkedPrefill 使用
            self.key_cache = key_cache    # int8
            self.value_cache = value_cache  # int8

        # ====================================================================
        # 【原有代码 - 与 w8a8.py 完全相同】
        # 路径1: PrefillNoCache - 预填充阶段，无 cache 命中
        # ====================================================================
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            """
            场景：第一次推理，cache 是空的，直接用当前的 key/value 计算
            特点：不需要反量化（因为用的是浮点的 key/value，不是 cache）
            vs AscendC8KVCacheMethod: 完全相同
            """
            assert attn_metadata.attn_mask is not None
            mask = attn_metadata.attn_mask  # attention mask
            
            # NPU 算子：torch_npu._npu_flash_attention
            # 输入：query/key/value 都是 bfloat16/float16
            # 输出：bfloat16/float16
            torch_npu._npu_flash_attention(
                query=query,                          # [num_tokens, num_heads, head_size], bfloat16/float16
                key=key,                              # [num_tokens, num_kv_heads, head_size], bfloat16/float16
                value=value,                          # [num_tokens, num_kv_heads, head_size], bfloat16/float16
                mask=mask,                            # attention mask
                seq_len=attn_metadata.seq_lens,       # 序列长度
                scale_value=scale,                    # 缩放因子 (1/sqrt(head_dim))
                num_heads=layer.num_heads,            # Q heads 数量
                num_kv_heads=layer.num_kv_heads,      # KV heads 数量
                out=output.reshape(query.shape)       # 输出：[num_tokens, num_heads, head_size], bfloat16/float16
            )

        # ====================================================================
        # 【我们新增的代码】
        # 路径2: PrefillCacheHit - 预填充阶段，有 cache 命中（Prefix Cache）
        # ====================================================================
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            """
            场景：使用 prefix cache，部分 KV 已经在 cache 中
            特点：需要显式反量化 cache 中的 int8 数据
            vs AscendC8KVCacheMethod: 原版直接 raise NotImplementedError
            """
            assert attn_metadata is not None
            assert attn_metadata.attn_mask is not None

            compress_mask = attn_metadata.attn_mask
            batch_size = attn_metadata.query_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]  # block 索引表
            num_block, block_size, _ = self.key_cache.shape  # type: ignore

            # 从 cache 中读取 key/value（都是 int8）
            key_from_cache = self.key_cache.view(num_block, block_size, -1)    # int8
            value_from_cache = self.value_cache.view(num_block, block_size, -1)  # int8

            # 【核心新增】反量化：int8 -> bfloat16/float16
            if key_from_cache.dtype == torch.int8:
                key_cache_anti_quant, value_cache_anti_quant = self.anti_quant_int8(
                    key_from_cache, value_from_cache, layer)
                # 输出：bfloat16/float16
            else:
                # 如果不是 int8（不应该发生），直接使用
                key_cache_anti_quant = key_from_cache
                value_cache_anti_quant = value_from_cache

            # 【新增】获取最大序列长度，用于选择合适的算子
            max_seq_len = max(attn_metadata.seq_lens_list) if hasattr(
                attn_metadata, 'seq_lens_list'
            ) and attn_metadata.seq_lens_list is not None else 0

            # 【新增】根据 block_size 和序列长度选择算子
            if block_size == 128 and max_seq_len <= 2048:
                # 使用 npu_fused_infer_attention_score (sparse_mode=3 有长度限制)
                key = key_cache_anti_quant.view(num_block, block_size, -1)    # bfloat16/float16
                value = value_cache_anti_quant.view(num_block, block_size, -1)  # bfloat16/float16

                # NPU 算子：torch_npu.npu_fused_infer_attention_score
                # 输入：query/key/value 都是 bfloat16/float16（反量化后）
                # 输出：bfloat16/float16
                output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,                               # bfloat16/float16
                    key=key,                                   # bfloat16/float16 (反量化后)
                    value=value,                               # bfloat16/float16 (反量化后)
                    atten_mask=compress_mask,
                    block_table=block_table,
                    input_layout="TND",                        # Token-Num_heads-Dim
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                    num_key_value_heads=layer.num_kv_heads,
                    num_heads=layer.num_heads,
                    scale=scale,
                    sparse_mode=3,
                )  # 输出：bfloat16/float16
            else:
                # 使用 _npu_flash_attention_qlens (支持更长的序列)
                # 输入/输出：bfloat16/float16
                torch_npu._npu_flash_attention_qlens(
                    query=query,                        # bfloat16/float16
                    key_cache=key_cache_anti_quant,     # bfloat16/float16 (反量化后)
                    value_cache=value_cache_anti_quant, # bfloat16/float16 (反量化后)
                    block_table=block_table,
                    mask=compress_mask,
                    seq_len=attn_metadata.query_lens,
                    context_lens=attn_metadata.seq_lens,
                    num_kv_heads=layer.num_kv_heads,
                    num_heads=layer.num_heads,
                    scale_value=scale,
                    out=output)  # 输出：bfloat16/float16

        # ====================================================================
        # 【原有代码 - 与 w8a8.py 完全相同】
        # 路径3: DecodeOnly - 解码阶段（增量解码）
        # ====================================================================
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            """
            场景：每次只生成一个 token
            特点：算子内部自动反量化（传入 antiquant_scale 参数）
            vs AscendC8KVCacheMethod: 完全相同
            """
            # 获取序列长度信息
            if hasattr(attn_metadata, "decode"):
                decode_meta = attn_metadata.decode
                seq_lens = decode_meta.seq_lens_list
            else:
                seq_lens = attn_metadata.seq_lens

            block_size = key_cache.shape[1]
            # Reshape query: [batch_size, 1, hidden_size]
            query = query.view(num_tokens, 1,
                               layer.num_heads * layer.head_size).contiguous()

            key = key_cache    # int8 (不需要提前反量化)
            value = value_cache  # int8 (不需要提前反量化)

            # NPU 算子：torch_npu.npu_incre_flash_attention
            # 【关键】这个算子支持 int8 输入 + antiquant_scale 参数
            # 算子内部会自动进行反量化，无需我们手动调用 npu_anti_quant
            # 
            # 输入：
            #   - query: bfloat16/float16
            #   - key/value: int8（算子内部自动反量化）
            #   - antiquant_scale: bfloat16/float16, shape=[2, num_kv_heads * head_size]
            # 输出：bfloat16/float16
            output = torch_npu.npu_incre_flash_attention(
                query,                                      # bfloat16/float16
                key,                                        # int8 (算子内部会反量化)
                value,                                      # int8 (算子内部会反量化)
                num_key_value_heads=layer.num_kv_heads,
                num_heads=layer.num_heads,
                actual_seq_lengths=seq_lens,
                scale_value=scale,
                input_layout='BSH',                         # Batch-Seq-Hidden
                block_size=block_size,
                block_table=attn_metadata.block_tables,
                antiquant_scale=self.antiquant_scale_comb,  # [2, num_kv_heads * head_size], bfloat16/float16
                                                            # [0, :] = key_scale, [1, :] = value_scale
                                                            # 算子内部使用这个 scale 进行反量化
            )  # 输出：bfloat16/float16

        # ====================================================================
        # 【我们新增的代码】
        # 路径4: ChunkedPrefill - 分块预填充（长上下文）
        # ====================================================================
        elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            """
            场景：长上下文场景，将 prefill 分成多个 chunk
            特点：需要显式反量化
            vs AscendC8KVCacheMethod: 原版直接 raise NotImplementedError
            """
            # head_size=192 的模型暂不支持（某些算子限制）
            if layer.head_size == 192:
                raise NotImplementedError(
                    "KV cache int8 quantization is not implemented for head_size == 192"
                )

            assert attn_metadata is not None
            assert attn_metadata.attn_mask is not None

            # 【新增】310P 设备需要特殊的格式转换
            if get_ascend_device_type() == AscendDeviceType._310P:
                attn_metadata.attn_mask = \
                    torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(),
                                              ACL_FORMAT_FRACTAL_NZ)
                # 优化 H2D 传输：使用 pin_memory + non_blocking 避免 CPU 同步
                if attn_metadata.seq_lens.device.type == 'cpu':
                    attn_metadata.seq_lens = \
                        attn_metadata.seq_lens.pin_memory().to(
                            device=query.device, non_blocking=True)
                else:
                    attn_metadata.seq_lens = \
                        attn_metadata.seq_lens.to(device=query.device)

            # 从 cache 读取（int8）
            num_block, block_size, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(num_block, block_size, -1)    # int8
            value = self.value_cache.view(num_block, block_size, -1)  # int8

            # 【核心新增】反量化：int8 -> bfloat16/float16
            if key.dtype == torch.int8:
                key, value = self.anti_quant_int8(key, value, layer)
                # 输出：bfloat16/float16

            # NPU 算子：torch_npu.npu_fused_infer_attention_score
            # 输入：query/key/value 都是 bfloat16/float16（反量化后）
            # 输出：bfloat16/float16
            output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,                                # bfloat16/float16
                key=key,                                    # bfloat16/float16 (反量化后)
                value=value,                                # bfloat16/float16 (反量化后)
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
            )  # 输出：bfloat16/float16
        
        # ====================================================================
        # 【原有代码】其他未实现的 attention state
        # ====================================================================
        else:
            raise NotImplementedError(
                f"KV cache int8 quantization is not implemented for other attention states: {attn_metadata.attn_state}"
            )

        return output  # bfloat16/float16, shape=[num_tokens, num_heads * head_size]
