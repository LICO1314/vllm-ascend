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
Qwen2/Qwen3 模型的 KV Cache 量化参数加载补丁

问题背景：
- 量化模型的参数名：k_proj.kv_cache_scale, v_proj.kv_cache_scale
- vLLM Qwen2Model 使用 fused qkv_proj，期望的参数名：attn.key_antiquant_scale, attn.value_antiquant_scale
- 需要在权重加载时进行参数名重映射

解决方案：
- 通过 Monkey Patch 替换 Qwen2Model.load_weights 方法
- 自动将量化模型的 kv_cache_scale 参数重映射到正确的名称
- 支持 Tensor Parallel 多卡并行，正确分片 scale 参数
"""

# ============================================================================
# 导入部分 - 标准导入，无特殊修改
# ============================================================================
from collections.abc import Iterable

import torch
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import is_pp_missing_parameter


# ============================================================================
# 【我们新增的代码】辅助函数：KV cache scale 参数加载
# ============================================================================
def _load_kv_cache_scale(name: str, loaded_weight: torch.Tensor,
                         params_dict: dict, loaded_params: set,
                         old_suffix: str, new_suffix: str) -> bool:
    """
    辅助函数：加载 KV cache scale 参数并进行 Tensor Parallel 分片
    
    这个函数是我们新写的，用于处理 C8 量化特有的参数重映射需求。
    
    核心功能：
    1. 参数名重映射：old_suffix -> new_suffix
       例如：k_proj.kv_cache_scale -> attn.key_antiquant_scale
    2. Tensor Parallel 分片：如果是多卡推理，只加载本卡对应的分片
    3. dtype 转换：将 float32 转换为模型需要的 dtype（通常是 bfloat16）
    
    参数：
        name: 当前权重的参数名（来自量化模型）
        loaded_weight: 加载的权重 tensor
            - shape: [total_num_kv_heads * head_size]
            - dtype: float32（量化模型通常使用 float32）
        params_dict: 模型中所有参数的字典
        loaded_params: 已加载参数的集合（用于记录）
        old_suffix: 需要替换的旧后缀（例如 "k_proj.kv_cache_scale"）
        new_suffix: 替换后的新后缀（例如 "attn.key_antiquant_scale"）
    
    返回：
        bool: 如果参数名匹配并成功加载返回 True，否则返回 False
    """
    # 步骤1: 检查参数名是否以 old_suffix 结尾
    if not name.endswith(old_suffix):
        return False

    # 步骤2: 进行参数名重映射
    # 例如：model.layers.0.self_attn.k_proj.kv_cache_scale
    #   -> model.layers.0.self_attn.attn.key_antiquant_scale
    remapped_name = name.replace(old_suffix, new_suffix)
    
    # 步骤3: 检查重映射后的参数名是否存在于模型中
    if remapped_name in params_dict:
        param = params_dict[remapped_name]  # 获取模型中对应的参数对象
        
        # 步骤4: 获取 Tensor Parallel 信息
        tp_size = get_tensor_model_parallel_world_size()  # 总共有多少张卡
        tp_rank = get_tensor_model_parallel_rank()        # 当前是第几张卡（从0开始）

        # 步骤5: 如果是多卡并行，进行分片
        if tp_size > 1:
            # 计算每张卡应该加载的分片大小
            # 例如：loaded_weight.shape[0] = 1024, tp_size = 2
            #      则 shard_size = 512，每张卡加载 512 个元素
            shard_size = loaded_weight.shape[0] // tp_size
            start_idx = tp_rank * shard_size      # 本卡分片的起始索引
            end_idx = (tp_rank + 1) * shard_size  # 本卡分片的结束索引
            loaded_weight = loaded_weight[start_idx:end_idx]
            
            # 示例：tp_size=2, tp_rank=0: loaded_weight[0:512]
            #       tp_size=2, tp_rank=1: loaded_weight[512:1024]

        # 步骤6: dtype 转换
        # 量化模型通常使用 float32 存储 scale，但运行时需要 bfloat16/float16
        if loaded_weight.dtype != param.dtype:
            loaded_weight = loaded_weight.to(param.dtype)
            # 输入 dtype: float32
            # 输出 dtype: bfloat16 或 float16（取决于模型配置）

        # 步骤7: 使用权重加载器加载参数
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        
        # 步骤8: 记录已加载的参数
        loaded_params.add(remapped_name)
    
    return True  # 参数名匹配，处理完成


# ============================================================================
# 【我们新增的代码】修改后的权重加载函数
# 基于 vLLM 原始 Qwen2Model.load_weights，添加了 KV cache scale 重映射逻辑
# ============================================================================
def qwen2_load_weights_with_kv_cache_remap(
        self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """
    Qwen2/Qwen3 模型的权重加载方法（替换原始方法）
    
    对比原版 vLLM 的 Qwen2Model.load_weights (@qwen2.py 第379-442行)：
    【新增】支持 KV cache scale 参数的重映射 (见下方代码块1)
    【保留】原有的 fused 模块处理逻辑 (见下方代码块2、3)
    
    核心功能：
    1. 【新增】处理 KV cache scale 参数的重映射
    2. 【原有】处理 fused 模块（qkv_proj, gate_up_proj）的权重加载
    3. 【原有】支持 Tensor Parallel 和 Pipeline Parallel
    
    参数：
        self: Qwen2Model 实例
        weights: 迭代器，每个元素是 (参数名, 权重tensor) 的元组
    
    返回：
        set[str]: 已加载的参数名集合
    """
    # ========================================================================
    # 【原有代码】fused 模块映射关系（来自 vLLM 原版）
    # ========================================================================
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),      # qkv_proj 包含 q, k, v 三个 projection
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),  # gate_up_proj 包含 gate 和 up 两个 projection
        ("gate_up_proj", "up_proj", 1),
    ]
    
    # 【原有代码】获取模型中所有参数的字典
    params_dict = dict(self.named_parameters(remove_duplicate=False))
    loaded_params: set[str] = set()  # 记录已加载的参数

    # 【原有代码】遍历所有待加载的权重
    for name, loaded_weight in weights:
        # ====================================================================
        # 【原有代码】跳过 RoPE 的 inv_freq 参数
        # ====================================================================
        if "rotary_emb.inv_freq" in name:
            continue

        # ====================================================================
        # 【代码块1 - 我们新增】处理 KV cache scale 参数重映射
        # 原版 vLLM 没有这部分代码，这是我们为了支持 C8 量化新增的
        # ====================================================================
        # 2. 处理 key 的 KV cache scale 参数
        # k_proj.kv_cache_scale -> attn.key_antiquant_scale
        if _load_kv_cache_scale(name, loaded_weight, params_dict,
                                loaded_params, "k_proj.kv_cache_scale",
                                "attn.key_antiquant_scale"):
            continue  # 已处理，跳到下一个参数

        # 3. 处理 value 的 KV cache scale 参数
        # v_proj.kv_cache_scale -> attn.value_antiquant_scale
        if _load_kv_cache_scale(name, loaded_weight, params_dict,
                                loaded_params, "v_proj.kv_cache_scale",
                                "attn.value_antiquant_scale"):
            continue  # 已处理，跳到下一个参数

        # 4. 【我们新增】跳过 kv_cache_offset 参数（C8 量化不需要 offset）
        if name.endswith("k_proj.kv_cache_offset") or name.endswith(
                "v_proj.kv_cache_offset"):
            continue
        # ====================================================================
        # 【代码块1 结束】
        # ====================================================================

        # ====================================================================
        # 【代码块2 - 原有代码】检查 Pipeline Parallel 缺失参数
        # ====================================================================
        # 在 PP 模式下，每个 stage 只需要加载部分层的参数
        if is_pp_missing_parameter(name, self):
            continue

        # ====================================================================
        # 【代码块3 - 原有代码】Fused 模块参数处理
        # 这部分逻辑来自 vLLM 原版，用于处理 qkv_proj 和 gate_up_proj 的权重加载
        # ====================================================================
        # 尝试匹配 stacked_params_mapping 中的模式
        for (param_name, weight_name, shard_id) in stacked_params_mapping:
            # 检查参数名是否包含原始权重名（例如 "q_proj", "k_proj"）
            if weight_name not in name:
                continue
            
            # 将原始名称替换为 fused 名称
            # 例如：self_attn.q_proj.weight -> self_attn.qkv_proj.weight
            name = name.replace(weight_name, param_name)
            
            # 【原有】跳过不存在的 bias 参数（某些量化方法会移除 bias）
            if name.endswith(".bias") and name not in params_dict:
                continue
            
            # 【原有】再次检查 Pipeline Parallel
            if is_pp_missing_parameter(name, self):
                continue
            
            # 【原有】处理 FP8 KV scale 的重映射（如果有）
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            
            # 【原有】加载权重
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            
            # 【原有】根据是否有自定义 weight_loader 决定调用方式
            if weight_loader == default_weight_loader:
                weight_loader(param, loaded_weight)
            else:
                # 自定义 weight_loader 需要 shard_id 来知道这是 q/k/v 中的哪一个
                weight_loader(param, loaded_weight, shard_id)
            break  # 找到匹配的 mapping，跳出循环
        
        # ====================================================================
        # 【代码块4 - 原有代码】普通参数处理（没有找到 fused mapping）
        # ====================================================================
        else:  # 如果上面的 for 循环没有 break（即没有找到匹配的 mapping）
            # 【原有】跳过不存在的 bias
            if name.endswith(".bias") and name not in params_dict:
                continue
            
            # 【原有】处理 FP8 KV scale 重映射
            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None:
                continue
            
            # 【原有】检查 Pipeline Parallel
            if is_pp_missing_parameter(name, self):
                continue
            
            # 【原有】加载权重
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
        
        # 【原有】记录已加载的参数
        loaded_params.add(name)
    
    return loaded_params


# ============================================================================
# 【我们新增的代码】Monkey Patch - 替换 Qwen2Model 的 load_weights 方法
# ============================================================================
# 在模块导入时，自动替换 Qwen2Model 的 load_weights 方法
# 这样所有使用 Qwen2Model 的地方都会使用我们的新实现
Qwen2Model.load_weights = qwen2_load_weights_with_kv_cache_remap
