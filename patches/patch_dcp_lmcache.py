#!/usr/bin/env python3
"""
DCP-aware LMCache patch — enables LMCache MP connector with DCP > 1.

Root causes fixed:
  1. Cache key collision: extract_world_size_and_kv_rank ignores DCP,
     so all DCP ranks get kv_rank=0 → same ObjectKey → silent corruption.
     FIX: For MLA + DCP>1, set kv_world_size=dcp_size, kv_rank=rank%dcp_size.

  2. compute_extra_count uses tp_size-1 for MLA multi-reader locking,
     but with DCP the TP group is split into tp_size/dcp_size DCP subgroups.
     FIX: (tp_size // world_size) - 1 instead of tp_size - 1.

  3. vllm_block_size is raw block_size, but scheduler uses block_size*dcp.
     FIX: vllm_block_size = block_size * dcp_size in adapter construction.

Patches both the LMCache package (primary) and vLLM builtin (fallback).
"""
import os
import re
import sys
import traceback
from pathlib import Path

PATCH_MARKER = "DCP-LMCACHE-PATCH"
VENV = "/opt/venv/lib/python3.12/site-packages"

# ─── helpers ──────────────────────────────────────────────────────────

def patch_file(path: str, old: str, new: str, check: str = None) -> bool:
    """Replace old with new in file. Returns True if patched."""
    p = Path(path)
    if not p.exists():
        print(f"  SKIP (not found): {path}")
        return False
    content = p.read_text()
    if new in content:
        print(f"  SKIP (already patched): {path}")
        return True
    if old not in content:
        print(f"  SKIP (pattern not found): {path}")
        return False
    content = content.replace(old, new)
    p.write_text(content)
    print(f"  OK: {path}")
    return True


def patch_extract_world_size_and_kv_rank(path: str) -> bool:
    """Patch extract_world_size_and_kv_rank to include DCP awareness."""
    p = Path(path)
    if not p.exists():
        print(f"  SKIP (not found): {path}")
        return False
    content = p.read_text()
    if f"{PATCH_MARKER}" in content and "dcp_size" in content:
        print(f"  SKIP (already patched): {path}")
        return True

    # Match the function body: the else branch that returns world_size//tp_size
    # This pattern works for both vLLM and LMCache copies
    old = """        # Tensor parallel does not change the KV caches for MLA models.
        # So we need to "exclude" the effect of TP on rank and world size
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        # vLLM constructs TP groups first, and then construct other
        # parallel groups on top of TP groups.
        # for example, TP=4, PP=2,
        # PP group: [0, 1, 2, 3], [4, 5, 6, 7]
        # TP group: [0, 4], [1, 5], [2, 6], [3, 7]
        # So we can "exclude" the effect of TP by rank // tp_size.
        return world_size // tp_size, rank // tp_size"""

    new = """        # Tensor parallel does not change the KV caches for MLA models.
        # So we need to "exclude" the effect of TP on rank and world size
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        # vLLM constructs TP groups first, and then construct other
        # parallel groups on top of TP groups.
        # for example, TP=4, PP=2,
        # PP group: [0, 1, 2, 3], [4, 5, 6, 7]
        # TP group: [0, 4], [1, 5], [2, 6], [3, 7]
        # So we can "exclude" the effect of TP by rank // tp_size.
        kv_world_size = world_size // tp_size
        kv_rank = rank // tp_size
        # {PATCH_MARKER}: DCP awareness for MLA
        # With DCP>1, the KV cache is sharded across dcp_size ranks.
        # For MLA, TP doesn't shard KV, so each DCP rank has a unique shard.
        # DCP groups are formed by reshape(-1, dcp_size), so rank%dcp_size
        # gives the DCP rank within the group (same as get_dcp_group().rank_in_group).
        dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_size > 1:
            kv_world_size = dcp_size
            kv_rank = rank % dcp_size
        return kv_world_size, kv_rank"""

    if old not in content:
        print(f"  SKIP (extract pattern not found): {path}")
        return False
    content = content.replace(old, new)
    p.write_text(content)
    print(f"  OK (extract_world_size_and_kv_rank): {path}")
    return True


def patch_vllm_block_size(path: str) -> bool:
    """Patch create_scheduler_adapter and create_worker_adapter to use
    block_size * dcp_size as vllm_block_size, AND patch the connector class's
    self.vllm_block_size assignment."""
    p = Path(path)
    if not p.exists():
        print(f"  SKIP (not found): {path}")
        return False
    content = p.read_text()
    if f"{PATCH_MARKER}" in content and "block_size * dcp" in content:
        # Check if connector self.vllm_block_size is also patched
        if "self.vllm_block_size = vllm_config.cache_config.block_size * " not in content:
            pass  # Need to patch connector class too
        else:
            print(f"  SKIP (already patched): {path}")
            return True

    changed = False

    # 1. Adapter construction: vllm_block_size=block_size → block_size * dcp
    old_adapter = "vllm_block_size=vllm_config.cache_config.block_size,"
    new_adapter = """vllm_block_size=vllm_config.cache_config.block_size * vllm_config.parallel_config.decode_context_parallel_size,  # {PATCH_MARKER}: DCP-aware scheduler block size"""

    if old_adapter in content:
        count = content.count(old_adapter)
        content = content.replace(old_adapter, new_adapter)
        print(f"  OK (adapter vllm_block_size, {count} occurrences)")
        changed = True

    # 2. Connector class: self.vllm_block_size = block_size → block_size * dcp
    old_conn = "self.vllm_block_size = vllm_config.cache_config.block_size"
    new_conn = "self.vllm_block_size = vllm_config.cache_config.block_size * vllm_config.parallel_config.decode_context_parallel_size  # {PATCH_MARKER}: DCP-aware connector block size"

    if old_conn in content:
        content = content.replace(old_conn, new_conn)
        print(f"  OK (connector self.vllm_block_size)")
        changed = True

    if changed:
        p.write_text(content)
    else:
        print(f"  SKIP (no patterns found)")
    return changed


def patch_compute_extra_count(path: str) -> bool:
    """Patch compute_extra_count to use (tp_size // world_size) - 1."""
    p = Path(path)
    if not p.exists():
        print(f"  SKIP (not found): {path}")
        return False
    content = p.read_text()
    if f"{PATCH_MARKER}" in content and "// world_size" in content:
        print(f"  SKIP (already patched): {path}")
        return True

    # Current: return tp - 1 if tp > world_size else 0
    # New: return (tp // world_size) - 1 if tp > world_size else 0
    old = "    return tp - 1 if tp > world_size else 0"
    new = f"    return (tp // world_size) - 1 if tp > world_size else 0  # {PATCH_MARKER}: DCP-aware extra_count"

    if old not in content:
        print(f"  SKIP (compute_extra_count pattern not found): {path}")
        return False
    content = content.replace(old, new)
    p.write_text(content)
    print(f"  OK (compute_extra_count): {path}")
    return True


# ─── main ─────────────────────────────────────────────────────────────

def main():
    print(f"[{PATCH_MARKER}] Applying DCP-aware LMCache patches...")
    print()

    # 1. extract_world_size_and_kv_rank — patch BOTH copies
    print("1. Patching extract_world_size_and_kv_rank:")
    targets = [
        f"{VENV}/lmcache/integration/vllm/lmcache_mp_connector.py",
        f"{VENV}/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py",
    ]
    for t in targets:
        patch_extract_world_size_and_kv_rank(t)

    # 2. vllm_block_size — patch BOTH copies
    print("\n2. Patching vllm_block_size (block_size * dcp):")
    for t in targets:
        patch_vllm_block_size(t)

    # 3. compute_extra_count — patch lookup module
    print("\n3. Patching compute_extra_count:")
    patch_compute_extra_count(
        f"{VENV}/lmcache/v1/multiprocess/modules/lookup.py"
    )

    # Also patch the backup if it exists (won't be used but keeps things consistent)
    bak = f"{VENV}/lmcache/v1/multiprocess/modules/lookup.py.bak_session_ttl_v1"
    if Path(bak).exists():
        patch_compute_extra_count(bak)

    print()
    print(f"[{PATCH_MARKER}] Done. Summary:")
    print("  - extract_world_size_and_kv_rank: MLA+DCP>1 → kv_world_size=dcp_size, kv_rank=rank%dcp_size")
    print("  - vllm_block_size: block_size * dcp_size (scheduler block units)")
    print("  - compute_extra_count: (tp_size // world_size) - 1 (DCP-aware TP sharing)")
    print()
    print("  WARNING: chunk_size must be >= block_size * dcp_size")
    print("  WARNING: existing L1/L2 cache is invalidated (different kv_rank bitmap)")


if __name__ == "__main__":
    main()
