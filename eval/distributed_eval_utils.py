from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedBatchPlan:
    local_item_counts: tuple[int, ...]
    sync_batches: int


def distributed_batch_plan(
    *,
    total_items: int,
    num_processes: int,
    batch_size: int,
) -> DistributedBatchPlan:
    if num_processes < 1:
        raise ValueError("num_processes must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    items_per_proc, extra_items = divmod(total_items, num_processes)
    local_item_counts = tuple(
        items_per_proc + (1 if process_index < extra_items else 0)
        for process_index in range(num_processes)
    )
    max_local_items = max(local_item_counts, default=0)
    sync_batches = (max_local_items + batch_size - 1) // batch_size
    return DistributedBatchPlan(local_item_counts, sync_batches)
