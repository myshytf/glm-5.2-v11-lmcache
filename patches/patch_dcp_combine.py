#!/usr/bin/env python3
"""
Patch B12X MLA sparse attention backend to add DCP LSE combine.

Root cause: B12xMLASparseImpl.forward_mqa() returns (out, lse) when
need_to_return_lse_for_decode=True (DCP>1), but never calls dcp_combine
to merge partial attention outputs across DCP ranks. Each GPU only sees
its 1/DCP_SIZE KV shard → incorrect attention → broken tool calls.

Fix: Add dcp_a2a_lse_reduce() call after local attention, matching the
flashinfer backend pattern.
"""
import re
from pathlib import Path

PATCH_MARKER = "DCP-COMBINE-PATCH"
FILE = "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py"


def main():
    p = Path(FILE)
    if not p.exists():
        print(f"[{PATCH_MARKER}] ERROR: {FILE} not found")
        return False

    content = p.read_text()

    if PATCH_MARKER in content:
        print(f"[{PATCH_MARKER}] Already patched, skipping.")
        return True

    # ── 1. Add imports at the top ──────────────────────────────────
    old_import_anchor = "from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens"
    new_import = """from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
# {PATCH_MARKER}: DCP LSE combine
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce""".format(PATCH_MARKER=PATCH_MARKER)

    if old_import_anchor not in content:
        print(f"[{PATCH_MARKER}] ERROR: import anchor not found")
        return False
    content = content.replace(old_import_anchor, new_import, 1)

    # ── 2. Patch decode path ───────────────────────────────────────
    # Current: returns (out, lse) directly from decode kernel
    old_decode = """            if self.need_to_return_lse_for_decode:
                return cast(
                    tuple[torch.Tensor, torch.Tensor],
                    self._sparse_mla_decode_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        forced_num_splits=self._num_splits_cap,
                        return_lse=True,
                        lse_scale="natural",
                    ),
                )"""

    new_decode = """            if self.need_to_return_lse_for_decode:
                # {PATCH_MARKER}: DCP LSE combine — merge partial attention
                # across DCP ranks before returning. Without this, each GPU
                # only uses its local KV shard → incorrect attention output.
                _dec_out, _dec_lse = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    self._sparse_mla_decode_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        forced_num_splits=self._num_splits_cap,
                        return_lse=True,
                        lse_scale="natural",
                    ),
                )
                _dec_out = dcp_a2a_lse_reduce(
                    _dec_out, _dec_lse, self._dcp_group, is_lse_base_on_e=False
                )
                return _dec_out, None""".format(PATCH_MARKER=PATCH_MARKER)

    if old_decode not in content:
        print(f"[{PATCH_MARKER}] ERROR: decode path pattern not found")
        return False
    content = content.replace(old_decode, new_decode, 1)

    # ── 3. Patch extend/prefill path ───────────────────────────────
    old_extend = """            lse = None
            if self.need_to_return_lse_for_decode:
                out, lse = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    self._sparse_mla_extend_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        return_lse=True,
                        lse_scale="natural",
                    ),
                )"""

    new_extend = """            lse = None
            if self.need_to_return_lse_for_decode:
                # {PATCH_MARKER}: DCP LSE combine for extend/prefill path
                out, lse = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    self._sparse_mla_extend_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        return_lse=True,
                        lse_scale="natural",
                    ),
                )
                out = dcp_a2a_lse_reduce(
                    out, lse, self._dcp_group, is_lse_base_on_e=False
                )
                lse = None""".format(PATCH_MARKER=PATCH_MARKER)

    if old_extend not in content:
        print(f"[{PATCH_MARKER}] ERROR: extend path pattern not found")
        return False
    content = content.replace(old_extend, new_extend, 1)

    # ── 4. Store dcp_group reference in __init__ ───────────────────
    # Add self._dcp_group near the dcp_rank assignment
    old_dcp_init = """        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group"""

    new_dcp_init = """        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self._dcp_group = get_dcp_group()  # {PATCH_MARKER}
            self.dcp_rank = self._dcp_group.rank_in_group"""

    # This pattern appears TWICE (metadata builder + impl). Replace both.
    count = content.count(old_dcp_init)
    if count == 0:
        print(f"[{PATCH_MARKER}] WARNING: dcp_init pattern not found, trying alt")
        # Try the alternative pattern without get_dcp_group (it's already imported)
        old_dcp_init2 = """        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group"""
        count2 = content.count(old_dcp_init2)
        print(f"  alt pattern count: {count2}")
    else:
        content = content.replace(old_dcp_init, new_dcp_init)
        print(f"  Patched {count} dcp_init occurrences")

    # Also handle the second occurrence in B12xMLASparseImpl (line ~550)
    old_dcp_init_impl = """        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group"""

    new_dcp_init_impl = """        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self._dcp_group = get_dcp_group()  # {PATCH_MARKER}
            self.dcp_rank = self._dcp_group.rank_in_group"""

    if old_dcp_init_impl in content:
        content = content.replace(old_dcp_init_impl, new_dcp_init_impl)
        print(f"  Patched impl dcp_init")

    # Handle the first occurrence in B12xMLASparseMetadataBuilder (line ~320)
    old_dcp_init_mb = """        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group"""

    new_dcp_init_mb = """        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self._dcp_group = get_dcp_group()  # {PATCH_MARKER}
            self.dcp_rank = self._dcp_group.rank_in_group"""

    if old_dcp_init_mb in content:
        content = content.replace(old_dcp_init_mb, new_dcp_init_mb)
        print(f"  Patched metadata builder dcp_init")

    # Need a fallback for _dcp_group when dcp_world_size == 1
    old_need_lse = """        self.need_to_return_lse_for_decode = (
            self.dcp_world_size > 1 and self.can_return_lse_for_decode
        )"""

    new_need_lse = """        # {PATCH_MARKER}: fallback group for DCP=1 (dcp_a2a_lse_reduce no-ops)
        if not hasattr(self, '_dcp_group'):
            from types import SimpleNamespace
            self._dcp_group = SimpleNamespace(world_size=1, device_group=None)
        self.need_to_return_lse_for_decode = (
            self.dcp_world_size > 1 and self.can_return_lse_for_decode
        )"""

    if old_need_lse in content:
        content = content.replace(old_need_lse, new_need_lse)
        print(f"  Patched need_to_return_lse with _dcp_group fallback")

    p.write_text(content)
    print(f"\n[{PATCH_MARKER}] Done. DCP LSE combine added to B12X MLA sparse.")
    print("  - Import: dcp_a2a_lse_reduce from dcp_alltoall")
    print("  - Decode path: combine partial attention via dcp_a2a_lse_reduce")
    print("  - Extend path: same combine")
    print("  - self._dcp_group stored in __init__ for both code paths")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
