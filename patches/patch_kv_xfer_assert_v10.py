#!/usr/bin/env python3
"""Idempotent startup source-patch for vLLM 0.11.2 (v10 image, /opt/venv install).

Guards BOTH asserts in the V1 scheduler's `_update_from_kv_xfer_finished` so a
late or duplicate LMCache KV-transfer completion callback for an already-freed
request (client abort / finish, or LMCacheMPConnector + kv_both +
async-scheduling returning a req_id twice — see LMCache #2356) is logged and
skipped instead of killing EngineCore with `AssertionError`.

Mirrors the fix on vLLM `main` (hard assert -> guarded skip). Runs before
`vllm serve`; edits the installed package source so every engine subprocess
imports the patched scheduler. Safe to run repeatedly; each sub-patch applies
independently (so a container restarted from a state where only the older
sending-only version of this script ran picks up just the missing recv guard).
"""
import sys

SCHED = "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"

# Patch 1 — finished_sending loop. The `assert` immediately followed by
# `_free_blocks` is unique to this loop.
SEND_OLD = (
    "            assert req_id in self.requests\n"
    "            self._free_blocks(self.requests[req_id])\n"
)
SEND_NEW = (
    "            if req_id not in self.requests:\n"
    "                logger.warning(\n"
    "                    \"Finished sending KV transfer for request %s, but it \"\n"
    "                    \"was already freed (LMCache abort/duplicate race; \"\n"
    "                    \"patched guard).\", req_id)\n"
    "            else:\n"
    "                self._free_blocks(self.requests[req_id])\n"
)
SEND_MARKER = "patched guard"

# Patch 2 — finished_recving loop. The `assert` immediately followed by
# `req = self.requests[req_id]` is unique to this loop (in both the pristine
# file and one where patch 1 is already applied).
RECV_OLD = (
    "            assert req_id in self.requests\n"
    "            req = self.requests[req_id]\n"
)
RECV_NEW = (
    "            if req_id not in self.requests:\n"
    "                logger.warning(\n"
    "                    \"Finished recving KV transfer for request %s, but it \"\n"
    "                    \"was already freed (LMCache abort/duplicate race; \"\n"
    "                    \"patched recv guard).\", req_id)\n"
    "                continue\n"
    "            req = self.requests[req_id]\n"
)
RECV_MARKER = "patched recv guard"


def apply(src: str, name: str, old: str, new: str, marker: str) -> str:
    if marker in src:
        print(f"[patch_kv_xfer_assert_v10] {name}: already applied; skipping")
        return src
    n = src.count(old)
    if n != 1:
        print(
            f"[patch_kv_xfer_assert_v10] ERROR: {name}: expected exactly 1 "
            f"match, found {n}. Source layout changed — refusing to patch "
            f"(would otherwise run the crash-prone version).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[patch_kv_xfer_assert_v10] applied {name} guard to scheduler.py")
    return src.replace(old, new, 1)


def main() -> int:
    try:
        with open(SCHED, "r") as f:
            src = f.read()
    except OSError as e:
        print(f"[patch_kv_xfer_assert_v10] ERROR: cannot read {SCHED}: {e}",
              file=sys.stderr)
        return 1

    src = apply(src, "finished_sending", SEND_OLD, SEND_NEW, SEND_MARKER)
    src = apply(src, "finished_recving", RECV_OLD, RECV_NEW, RECV_MARKER)

    with open(SCHED, "w") as f:
        f.write(src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
