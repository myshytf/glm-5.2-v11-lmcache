#!/usr/bin/env python3
"""
Safe preempt-restore patch for LMCacheMPConnector + LMCache session TTL.

v3 fixes the v2 bug where ret=None (async lookup still pending) was treated as
"no hit" and the one-shot guard permanently skipped future polling. v3 submits
once, then returns (None, True) while LMCache lookup/prefetch is pending so vLLM
scheduler polls again instead of recomputing immediately.

It also changes LMCache SessionManager.remove() to TTL-retain finished sessions
and invokes cleanup_expired() from end_session(), so end_session touch logic keeps
working without leaking sessions forever.
"""

from __future__ import annotations

import os
import py_compile
import re
import sys
from pathlib import Path

CONNECTOR_TARGETS = [
    Path("/opt/venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py"),
    Path("/opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/lmcache_mp_connector.py"),
]
SESSION_PATH = Path("/opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/session.py")
LOOKUP_PATH = Path("/opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/modules/lookup.py")

OLD_UPSTREAM = """        # TODO: support loading KV for preempted requests in the future
        if request.status == RequestStatus.PREEMPTED:
            return 0, False"""

SAFE_V3 = """        # [SAFE-PREEMPT-RESTORE v3] Async-aware one-shot LMCache restore.
        # Upstream returned (0, False) for PREEMPTED, forcing full recompute.
        # v2 submitted a lookup but treated ret=None (prefetch pending) as miss.
        # v3 submits once, then keeps returning (None, True) while pending so
        # the scheduler polls again and can actually receive the L1/L2 result.
        if request.status == RequestStatus.PREEMPTED:
            _rid = request.request_id
            _ctx_len = len(request.all_token_ids)
            if not hasattr(self, '_safe_preempt_submitted'):
                self._safe_preempt_submitted = set()
                self._safe_preempt_finished = set()
                self._safe_preempt_active = set()

            if _rid in self._safe_preempt_finished:
                return 0, False

            try:
                if _rid not in self._safe_preempt_submitted:
                    # Guard: only one preempt-restore lookup active at a time.
                    # This avoids scheduler-wide LMCache lookup storms under KV pressure.
                    if self._safe_preempt_active:
                        logger.info(
                            "SafePreemptRestore: %s skipped; active restore(s)=%s",
                            _rid, list(self._safe_preempt_active))
                        return 0, False
                    self.scheduler_adapter.maybe_submit_lookup_request(
                        request.request_id,
                        token_ids=list(request.all_token_ids),
                        cache_salt=tracker.cache_salt)
                    self._safe_preempt_submitted.add(_rid)
                    self._safe_preempt_active.add(_rid)
                    logger.info(
                        "SafePreemptRestore: %s submitted lookup (ctx=%d, salt=%r)",
                        _rid, _ctx_len, tracker.cache_salt)

                ret = self.scheduler_adapter.check_lookup_result(request.request_id)
            except Exception as e:
                self._safe_preempt_submitted.discard(_rid)
                self._safe_preempt_active.discard(_rid)
                self._safe_preempt_finished.add(_rid)
                logger.warning("SafePreemptRestore: %s lookup error: %s", _rid, e)
                return 0, False

            if ret is None:
                logger.info(
                    "SafePreemptRestore: %s lookup pending (ctx=%d); wait.",
                    _rid, _ctx_len)
                return None, True

            # Final result — clean active state and never try this request again.
            self._safe_preempt_submitted.discard(_rid)
            self._safe_preempt_active.discard(_rid)
            self._safe_preempt_finished.add(_rid)

            if ret == 0:
                logger.info(
                    "SafePreemptRestore: %s no hit (ctx=%d), recompute.",
                    _rid, _ctx_len)
                return 0, False

            try:
                _chunk = (self.scheduler_adapter.num_blocks_per_chunk()
                          * self.vllm_block_size)
                if ret % _chunk != 0:
                    logger.warning(
                        "SafePreemptRestore: %s misaligned ret=%d chunk=%d, recompute.",
                        _rid, ret, _chunk)
                    return 0, False
                _num_vllm = num_computed_tokens // self.vllm_block_size
                _num_lmc = ret // self.vllm_block_size
                tracker.increase_num_stored_blocks(_num_lmc)
                tracker.num_vllm_hit_blocks = _num_vllm
                tracker.num_lmcache_hit_blocks = _num_lmc
                _need = max(0, ret - num_computed_tokens)
                logger.info(
                    "SafePreemptRestore: %s RESTORED %d/%d tok from LMCache "
                    "(need_load=%d) — skip recompute!",
                    _rid, ret, _ctx_len, _need)
                return _need, _need > 0
            except Exception as e:
                logger.warning(
                    "SafePreemptRestore: %s process error: %s, recompute.",
                    _rid, e)
                return 0, False"""


def backup(path: Path, suffix: str) -> None:
    b = path.with_name(path.name + suffix)
    if not b.exists() and path.exists():
        b.write_text(path.read_text(errors="replace"))
        print(f"✅ backup {b}")


def compile_file(path: Path) -> None:
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        print(f"❌ syntax failed: {path}: {e}")
        sys.exit(1)


def patch_connector(path: Path) -> bool:
    if not path.exists():
        print(f"⚠️ missing connector {path}")
        return False
    text = path.read_text()
    if "[SAFE-PREEMPT-RESTORE v3]" in text:
        print(f"↪️ {path}: v3 already applied")
        return False
    backup(path, ".bak_safe_preempt_v3")
    original = text

    # Replace any previous safe preempt block (v2/v3 variants) up to the normal path.
    m = re.search(
        r"        # \[SAFE-PREEMPT-RESTORE v[0-9]+\].*?\n\n        self\.scheduler_adapter\.maybe_submit_lookup_request",
        text,
        flags=re.S,
    )
    if m:
        text = text[:m.start()] + SAFE_V3 + "\n\n        self.scheduler_adapter.maybe_submit_lookup_request" + text[m.end():]
    elif OLD_UPSTREAM in text:
        text = text.replace(OLD_UPSTREAM, SAFE_V3, 1)
    elif "attempting LMCache lookup for" in text:
        # Old v1 patch. Prefer the pristine backup if present.
        old_backup = path.with_name(path.name + ".bak_preempted")
        if old_backup.exists():
            text = old_backup.read_text()
            if OLD_UPSTREAM not in text:
                print(f"❌ {path}: v1 backup lacks upstream marker")
                return False
            text = text.replace(OLD_UPSTREAM, SAFE_V3, 1)
        else:
            print(f"❌ {path}: old v1 patch found but no backup to restore")
            return False
    else:
        print(f"❌ {path}: no known preempt block found")
        return False

    if text != original:
        path.write_text(text)
        compile_file(path)
        print(f"✅ {path}: applied SAFE-PREEMPT-RESTORE v3")
        return True
    return False


def patch_session_manager() -> bool:
    if not SESSION_PATH.exists():
        print(f"⚠️ missing session.py: {SESSION_PATH}")
        return False
    text = SESSION_PATH.read_text()
    if "[SESSION-TTL-RETAIN v1]" in text:
        print("↪️ session.py: TTL retain already applied")
        return False
    backup(SESSION_PATH, ".bak_session_ttl_v1")
    old = '''    def remove(self, request_id: str) -> Optional[Session]:
        """Remove a session by request_id.

        Args:
            request_id: Unique request identifier.

        Returns:
            The removed session, or None if no session was found.
        """
        with self._lock:
            if request_id in self._sessions:
                session = self._sessions[request_id]
                del self._sessions[request_id]
                logger.debug("Removed session for request_id=%s", request_id)
                return session
            return None
'''
    new = '''    def remove(self, request_id: str) -> Optional[Session]:
        """TTL-retain a session by request_id and return it.

        [SESSION-TTL-RETAIN v1] Upstream deleted sessions immediately on
        END_SESSION. That made later touch/restore paths lose request hash
        metadata during preempt/requeue races. Keep the session until the
        normal TTL cleanup instead; refresh created_at so TTL starts at finish.
        """
        with self._lock:
            session = self._sessions.get(request_id)
            if session is not None:
                session.created_at = time.time()
                logger.debug("TTL-retained session for request_id=%s", request_id)
                return session
            return None
'''
    if old not in text:
        print("❌ session.py: remove() target not found")
        return False
    SESSION_PATH.write_text(text.replace(old, new, 1))
    compile_file(SESSION_PATH)
    print("✅ session.py: remove() now TTL-retains")
    return True


def patch_lookup_cleanup() -> bool:
    if not LOOKUP_PATH.exists():
        print(f"⚠️ missing lookup.py: {LOOKUP_PATH}")
        return False
    text = LOOKUP_PATH.read_text()
    if "[SESSION-TTL-CLEANUP v1]" in text:
        print("↪️ lookup.py: TTL cleanup already applied")
        return False
    backup(LOOKUP_PATH, ".bak_session_ttl_v1")
    old = '''        # unified touch of all keys, which include retrieved and stored keys
        # TODO(chunxiaozheng): when l2 is enabled, the prefetched keys from l2 are temp
        #  and will be deleted after finish_read_prefetched, when we touch all keys,
        #  these keys has been deleted and will not be touched.
        self._ctx.storage_manager.touch_l1_keys(obj_keys)
'''
    new = '''        # unified touch of all keys, which include retrieved and stored keys
        # TODO(chunxiaozheng): when l2 is enabled, the prefetched keys from l2 are temp
        #  and will be deleted after finish_read_prefetched, when we touch all keys,
        #  these keys has been deleted and will not be touched.
        self._ctx.storage_manager.touch_l1_keys(obj_keys)

        # [SESSION-TTL-CLEANUP v1] remove() now retains finished sessions for
        # TTL-based reuse, so opportunistically clean expired sessions here.
        try:
            self._ctx.session_manager.cleanup_expired()
        except Exception:
            logger.exception("Session TTL cleanup failed")
'''
    if old not in text:
        print("❌ lookup.py: touch block target not found")
        return False
    LOOKUP_PATH.write_text(text.replace(old, new, 1))
    compile_file(LOOKUP_PATH)
    print("✅ lookup.py: end_session now runs TTL cleanup")
    return True


def main() -> None:
    changed = False
    for path in CONNECTOR_TARGETS:
        changed = patch_connector(path) or changed
    changed = patch_session_manager() or changed
    changed = patch_lookup_cleanup() or changed

    # hard verification
    actual = CONNECTOR_TARGETS[1]
    if actual.exists() and "[SAFE-PREEMPT-RESTORE v3]" not in actual.read_text():
        print("❌ actual LMCache connector missing v3 patch")
        sys.exit(2)
    if SESSION_PATH.exists() and "[SESSION-TTL-RETAIN v1]" not in SESSION_PATH.read_text():
        print("❌ Session TTL retain missing")
        sys.exit(3)
    print(f"\n📊 safe-preempt/session patch complete; changed={changed}")


if __name__ == "__main__":
    main()
