from eval.distributed_eval_utils import distributed_batch_plan


def test_distributed_batch_plan_keeps_uneven_ranks_in_final_sync() -> None:
    plan = distributed_batch_plan(total_items=2500, num_processes=8, batch_size=1)

    assert plan.local_item_counts == (313, 313, 313, 313, 312, 312, 312, 312)
    assert plan.sync_batches == 313


def test_distributed_batch_plan_handles_batched_uneven_shards() -> None:
    plan = distributed_batch_plan(total_items=10, num_processes=3, batch_size=2)

    assert plan.local_item_counts == (4, 3, 3)
    assert plan.sync_batches == 2
