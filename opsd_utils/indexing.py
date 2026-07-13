from __future__ import annotations


def source_row_index(
    row: int,
    *,
    raw_count: int,
    expanded_count: int,
    num_generations: int,
) -> int:
    """Map a completion-level row to the source sample row.

    Some trainer paths receive batches already expanded by the repeat sampler
    (`raw_count == expanded_count`), while others may receive one row per prompt
    (`raw_count * num_generations == expanded_count`). Completion-level routing
    should use this mapping before reading sample fields or references.
    """
    if raw_count <= 0:
        return 0
    row = max(0, int(row))
    expanded_count = max(0, int(expanded_count))
    num_generations = max(1, int(num_generations))

    if raw_count == expanded_count:
        return min(row, raw_count - 1)
    if raw_count * num_generations == expanded_count:
        return min(row // num_generations, raw_count - 1)
    return min(row, raw_count - 1)
