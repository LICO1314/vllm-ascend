import threading
from typing import Iterable

import torch
import vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector as mc
from vllm.logger import logger


def _normalize_kv_cache_entry(entry: object) -> tuple[torch.Tensor, ...]:
    # Unwrap single-element containers to reach the actual entry.
    while isinstance(entry, (list, tuple)) and len(entry) == 1:
        entry = entry[0]
    if isinstance(entry, torch.Tensor):
        return (entry,)
    if isinstance(entry, (list, tuple)):
        if all(isinstance(x, torch.Tensor) for x in entry):
            return tuple(entry)
        if len(entry) > 0 and all(isinstance(x, (list, tuple)) for x in entry):
            return _normalize_kv_cache_entry(entry[0])
    raise TypeError(f"Unexpected kv_cache_entry type: {type(entry)}")


def _iter_caches(
    cache_tuple: tuple[torch.Tensor, ...],
    split_k_and_v: bool,
) -> Iterable[torch.Tensor]:
    return cache_tuple if split_k_and_v else (cache_tuple[0],)


def _patched_register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    """Register the KV Cache data in mooncake (patched for nested tuples)."""
    logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

    kv_data_ptrs = []
    kv_data_lens = []
    seen_base_addresses = []

    split_k_and_v = self.kv_topo.split_k_and_v
    tensor_size_bytes = None
    for layer_name, cache_or_caches in kv_caches.items():
        cache_tuple = _normalize_kv_cache_entry(cache_or_caches)
        logger.debug(
            "registering layer %s with shape %s", layer_name, cache_tuple[0].shape
        )
        for cache in _iter_caches(cache_tuple, split_k_and_v):
            base_addr = cache.data_ptr()
            if base_addr in seen_base_addresses:
                continue

            seen_base_addresses.append(base_addr)
            curr_tensor_size_bytes = cache.nbytes

            if tensor_size_bytes is None:
                tensor_size_bytes = curr_tensor_size_bytes
                self.num_blocks = cache.shape[0]

            assert tensor_size_bytes == curr_tensor_size_bytes, (
                "All kv cache tensors must have the same size"
            )
            kernel_block_size = cache.shape[-2 if self.use_mla else -3]
            assert self.block_size == kernel_block_size
            kv_data_ptrs.append(base_addr)
            kv_data_lens.append(tensor_size_bytes)

    self.kv_caches_base_addr = seen_base_addresses

    ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
    if ret_value != 0:
        raise RuntimeError("Mooncake batch memory registration failed.")

    assert tensor_size_bytes is not None
    assert self.num_blocks != 0
    assert tensor_size_bytes % self.num_blocks == 0
    self.block_len = tensor_size_bytes // self.num_blocks
    self.device_kv_caches = kv_caches
    logger.debug(
        "registered num_blocks=%d block_len=%d", self.num_blocks, self.block_len
    )

    # No need to launch server for D node.
    if self.kv_role == "kv_consumer":
        return

    ready_event = threading.Event()
    self._mooncake_sender_t = threading.Thread(
        target=self._mooncake_sender,
        args=(ready_event, self.side_channel_port, self.tp_rank),
        daemon=True,
        name="mooncake_sender",
    )
    self._mooncake_sender_t.start()
    ready_event.wait()  # Wait for listener ZMQ socket to be ready.


mc.MooncakeConnectorWorker.register_kv_caches = _patched_register_kv_caches
