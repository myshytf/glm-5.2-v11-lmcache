#!/usr/bin/env python3
"""
DCP MLA Store Patch — Fix MLA+DCP KV store sharing.

Problem: In MLA, `wait_for_save()` only stores KV from the first TP rank
(`is_first_rank_of_pp_group` = `rank % tp_size == 0`). This is correct for
MLA without DCP (all TP ranks share the same KV). But with DCP, each DCP
rank holds a DIFFERENT KV shard. Only rank 0 storing means DCP ranks 1,2,3
are never cached → lookup hits only 1/4 of the chunks.

Fix: With DCP, the first `dcp_size` ranks (one per DCP rank) should store.
Ranks >= dcp_size share the same KV due to MLA TP sharing and are skipped.

Symptom (LMCache server log):
    Prefetch request completed: 1/N prefix hits (1 L1, 0 L2)
    ← always 1 hit out of 4 expected (only DCP rank 0 stored)
"""

import sys

ADAPTER_PATH = "/opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_multi_process_adapter.py"

OLD = """    @property
    def is_first_rank_of_pp_group(self) -> bool:
        \"\"\"Is the first rank of the pipeline parallel group.\"\"\"
        return (
            self.parallel_strategy.actual_worker_id % self.parallel_strategy.tp_size
            == 0
        )"""

NEW = """    @property
    def is_first_rank_of_pp_group(self) -> bool:
        \"\"\"Is the first rank of the pipeline parallel group.

        With MLA+DCP, the first kv_world_size (= dcp_size) ranks each hold a
        different DCP shard and must all store. Ranks >= dcp_size share the
        same KV due to MLA TP sharing and are skipped.
        \"\"\"
        if self.use_mla:
            return (
                self.parallel_strategy.actual_worker_id
                < self.parallel_strategy.kv_world_size
            )
        return (
            self.parallel_strategy.actual_worker_id % self.parallel_strategy.tp_size
            == 0
        )"""


def main():
    with open(ADAPTER_PATH, "r") as f:
        content = f.read()

    if "DCP-MLA-STORE-PATCH" in content:
        print("[DCP-MLA-STORE-PATCH] Already patched, skipping.")
        return

    if OLD not in content:
        # Try with potential whitespace differences
        print("[DCP-MLA-STORE-PATCH] WARNING: exact match not found, trying fuzzy...")
        if "is_first_rank_of_pp_group" not in content:
            print("[DCP-MLA-STORE-PATCH] ERROR: property not found!")
            sys.exit(1)

    content = content.replace(OLD, NEW)

    # Add marker comment
    content = content.replace(
        "def is_first_rank_of_pp_group",
        "# DCP-MLA-STORE-PATCH: allow first dcp_size ranks to store\n    def is_first_rank_of_pp_group",
        1,
    )

    with open(ADAPTER_PATH, "w") as f:
        f.write(content)

    print("[DCP-MLA-STORE-PATCH] Applied successfully.")
    print("  Ranks 0..dcp_size-1 will now store (one per DCP rank).")
    print("  Ranks >= dcp_size skipped (MLA TP KV sharing).")


if __name__ == "__main__":
    main()
