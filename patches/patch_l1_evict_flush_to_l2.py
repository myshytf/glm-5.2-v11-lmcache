#!/usr/bin/env python3
"""
Patch LMCache 0.4.6: make L1 eviction preserve chunks in L2 without racing the
StoreController.

Previous patch used adapter.submit_store_task() then polled
pop_completed_store_tasks() from the eviction thread. That races with
StoreController, which owns the same completion queue, causing both:
  - "Completed store task ... not found in tracking"
  - "L1→L2 flush ... did not complete within timeout"

This version adds a synchronous FSL2Adapter.store_objects_sync() path and makes
L1EvictionController call it directly. No shared completion queue is consumed.
"""

from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

EVICT = Path("/opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/storage_controllers/eviction_controller.py")
SM = Path("/opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/storage_manager.py")
FS = Path("/opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py")

NEW_IMPORT_BLOCK = '''from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controller import StorageControllerInterface'''

SYNC_METHOD = r'''    def store_objects_sync(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> tuple[bool, int, int]:
        """Synchronously persist objects to FS L2 without using task queues.

        [SYNC-L2-STORE v1] Used by L1 eviction. The normal async store path is
        owned by StoreController; eviction must not consume its completion queue.

        Returns:
            (success, persisted_count, bytes_written). persisted_count includes
            keys already present on disk; bytes_written counts only new writes.
        """
        success = True
        bytes_written = 0
        persisted_count = 0
        newly_stored_keys: list[ObjectKey] = []
        newly_stored_sizes: list[int] = []
        touched_existing: list[ObjectKey] = []

        for key, obj in zip(keys, objects, strict=True):
            file_path, tmp_path = self._key_to_file_and_tmp_path(key)
            try:
                if file_path.exists():
                    persisted_count += 1
                    touched_existing.append(key)
                    continue

                buf = obj.byte_array
                size = len(buf)
                do_odirect = self._use_odirect
                if do_odirect:
                    aligned = self._os_disk_bs > 0 and size % self._os_disk_bs == 0
                    if not aligned:
                        logger.warning(
                            "Cannot use O_DIRECT for writing size %d, not aligned to block size %d.",
                            size, self._os_disk_bs)
                        do_odirect = False

                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                if do_odirect:
                    self._write_with_odirect(tmp_path, buf)
                else:
                    with open(tmp_path, "wb") as f:
                        f.write(buf)
                os.replace(tmp_path, file_path)

                persisted_count += 1
                bytes_written += size
                newly_stored_keys.append(key)
                newly_stored_sizes.append(size)
            except Exception:
                logger.exception("FSL2Adapter sync store failed for %s", file_path)
                success = False
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

        if newly_stored_keys:
            self._notify_keys_stored(newly_stored_keys, newly_stored_sizes)
        if touched_existing:
            self._notify_keys_accessed(touched_existing)
        return success, persisted_count, bytes_written

'''

NEW_L1_CLASS = '''class L1EvictionController(EvictionController):
    """
    Eviction controller for L1 cache.

    Patched behavior: when L2 adapters are available, L1 eviction actions use
    EvictionDestination.L2_CACHE and synchronously flush readable L1 objects to
    L2 before deleting them from L1. This prevents L1 eviction from silently
    dropping multi-turn KV chunks before the async StoreController can persist
    them to disk.
    """

    def __init__(
        self,
        l1_manager: L1Manager,
        eviction_config: EvictionConfig,
        l2_adapters: list[L2AdapterInterface] | None = None,
    ):
        super().__init__()
        self._eviction_config = eviction_config
        self._eviction_policy = CreateEvictionPolicy(eviction_config)
        self._l1_manager = l1_manager
        self._l2_adapters: list[L2AdapterInterface] | None = l2_adapters
        if self._l2_adapters:
            self._eviction_policy.register_eviction_destination(
                EvictionDestination.L2_CACHE
            )
        self._listener = L1EvictionPolicy(self._eviction_policy)
        self._l1_manager.register_listener(self._listener)
        self._event_bus = get_event_bus()

    def set_l2_adapters(self, l2_adapters: list[L2AdapterInterface]) -> None:
        """Late-inject L2 adapters after StorageManager creates them."""
        self._l2_adapters = l2_adapters
        self._eviction_policy.register_eviction_destination(
            EvictionDestination.L2_CACHE
        )

    def report_status(self) -> dict:
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "eviction_policy": self._eviction_config.eviction_policy,
            "trigger_watermark": self._eviction_config.trigger_watermark,
            "eviction_ratio": self._eviction_config.eviction_ratio,
            "l2_flush_enabled": bool(self._l2_adapters),
        }

    def _publish_skipped(self, usage: float, watermark: float) -> None:
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_EVICTION_LOOP_TICK,
                metadata={"usage": usage, "watermark": watermark, "triggered": False},
            )
        )

    def _publish_triggered(self, usage: float, watermark: float) -> None:
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_EVICTION_LOOP_TICK,
                metadata={"usage": usage, "watermark": watermark, "triggered": True},
            )
        )

    def eviction_loop(self):
        watermark = self._eviction_config.trigger_watermark
        eviction_ratio = self._eviction_config.eviction_ratio

        while not self._stop_flag.is_set():
            time.sleep(1)
            used_bytes, total_bytes = self._l1_manager.get_memory_usage()
            usage = 0 if total_bytes == 0 else used_bytes / total_bytes
            if usage < watermark:
                logger.debug(
                    "L1 memory usage %.2f below watermark %.2f; skipping eviction.",
                    usage, watermark)
                self._publish_skipped(usage, watermark)
                continue

            logger.info(
                "L1 memory usage %.2f above watermark %.2f; triggering eviction.",
                usage, watermark)
            actions = self._eviction_policy.get_eviction_actions(
                eviction_ratio,
                key_eligible_filter=self._l1_manager.is_key_evictable,
            )
            for action in actions:
                self.execute_eviction_action(action)
            self._publish_triggered(usage, watermark)

    def execute_eviction_action(self, action: EvictionAction):
        if action.destination == EvictionDestination.L2_CACHE and self._l2_adapters:
            self._flush_to_l2_then_delete(action.keys)
        elif action.destination == EvictionDestination.DISCARD:
            self._l1_manager.delete(action.keys)
        else:
            if action.destination != EvictionDestination.DISCARD:
                logger.error("Unsupported eviction destination: %s", action.destination)
                logger.error("Treating it as DISCARD.")
            self._l1_manager.delete(action.keys)

    def _flush_to_l2_then_delete(self, keys: list[ObjectKey]) -> None:
        """Synchronously persist readable L1 objects to L2, then delete from L1.

        [SYNC-L1-EVICT-FLUSH v2] Uses adapter.store_objects_sync() when present
        so eviction does not race StoreController's async completion queue.
        """
        if not keys:
            return

        read_result = self._l1_manager.reserve_read(keys)
        readable_keys: list[ObjectKey] = []
        readable_objs: list = []
        failed_keys: list[ObjectKey] = []

        for key in keys:
            entry = read_result.get(key)
            if entry is None or entry[0] != L1Error.SUCCESS or entry[1] is None:
                failed_keys.append(key)
            else:
                readable_keys.append(key)
                readable_objs.append(entry[1])

        flushed = False
        if readable_keys and self._l2_adapters:
            for idx, adapter in enumerate(self._l2_adapters):
                sync_store = getattr(adapter, "store_objects_sync", None)
                if sync_store is None:
                    logger.warning(
                        "L1→L2 flush: adapter %d has no sync store; preserving %d readable keys in L1",
                        idx, len(readable_keys))
                    continue
                try:
                    ok, persisted_count, bytes_written = sync_store(readable_keys, readable_objs)
                    if ok:
                        flushed = True
                        logger.info(
                            "L1→L2 flush: sync persisted %d/%d keys (%d bytes) via adapter %d",
                            persisted_count, len(readable_keys), bytes_written, idx)
                    else:
                        logger.warning(
                            "L1→L2 flush: adapter %d reported partial/failed sync persist (%d/%d keys, %d bytes)",
                            idx, persisted_count, len(readable_keys), bytes_written)
                except Exception:
                    logger.exception("L1→L2 flush: sync store failed on adapter %d", idx)

        if readable_keys:
            self._l1_manager.finish_read(readable_keys)

        # Delete unreadable/missing keys (they are not in L1 anymore). Delete readable
        # keys only after at least one L2 adapter persisted them successfully.
        keys_to_delete = list(failed_keys)
        if flushed:
            keys_to_delete.extend(readable_keys)
        elif readable_keys:
            logger.warning(
                "L1→L2 flush: no adapter persisted %d readable keys; keeping them in L1",
                len(readable_keys))

        if keys_to_delete:
            result = self._l1_manager.delete(keys_to_delete)
            not_deleted = [k for k, err in result.items() if err != L1Error.SUCCESS]
            if not_deleted:
                logger.debug("L1→L2 flush: %d keys not deleted, likely still locked", len(not_deleted))


'''


def backup(path: Path, suffix: str) -> None:
    b = path.with_name(path.name + suffix)
    if not b.exists() and path.exists():
        b.write_text(path.read_text(errors="replace"))
        print(f"✅ backup {b}")


def compile_file(path: Path) -> None:
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        print(f"❌ syntax check failed {path}: {e}")
        sys.exit(1)


def patch_fs_adapter() -> bool:
    text = FS.read_text()
    if "[SYNC-L2-STORE v1]" in text:
        print("↪️ fs_l2_adapter.py: sync store already applied")
        return False
    backup(FS, ".bak_sync_l2_store_v1")
    marker = "    # ---- store ----------------------------------------------------------\n"
    if marker not in text:
        print("❌ fs_l2_adapter.py: store marker not found")
        return False
    FS.write_text(text.replace(marker, SYNC_METHOD + marker, 1))
    compile_file(FS)
    print("✅ fs_l2_adapter.py: added store_objects_sync")
    return True


def patch_eviction_controller() -> bool:
    text = EVICT.read_text()
    original = text
    backup(EVICT, ".bak_l1_flush_v2")

    # Clean/ensure import block.
    text = re.sub(
        r"from lmcache\.v1\.distributed\.internal_api import \(.*?\n\)\n(?:from lmcache\.v1\.distributed\.error import L1Error\n)?(?:from lmcache\.v1\.distributed\.l1_manager import L1Manager\n)(?:from lmcache\.v1\.distributed\.l2_adapters\.base import L2AdapterInterface\n)?from lmcache\.v1\.distributed\.storage_controller import StorageControllerInterface",
        NEW_IMPORT_BLOCK,
        text,
        flags=re.S,
    )

    # Replace whole L1EvictionController up to next class.
    text, n = re.subn(
        r"class L1EvictionController\(EvictionController\):.*?\n\nclass L2AdapterEvictionState:",
        NEW_L1_CLASS + "class L2AdapterEvictionState:",
        text,
        count=1,
        flags=re.S,
    )
    if n != 1:
        print("❌ eviction_controller.py: L1EvictionController replacement failed")
        return False

    if text != original:
        EVICT.write_text(text)
        compile_file(EVICT)
        print("✅ eviction_controller.py: L1 eviction sync-flush v2 applied")
        return True
    print("↪️ eviction_controller.py: unchanged")
    return False


def patch_storage_manager() -> bool:
    text = SM.read_text()
    if "set_l2_adapters(self._l2_adapters)" in text:
        print("↪️ storage_manager.py: L2 adapter injection already applied")
        return False
    backup(SM, ".bak_l1_flush_v2")
    marker = '''        self._quota_manager = QuotaManager()\n\n        # Unified L2 eviction controller'''
    insert = '''        self._quota_manager = QuotaManager()\n\n        # Wire L2 adapters into L1 eviction controller so evicted chunks\n        # are flushed to L2 disk instead of silently discarded.\n        self._eviction_controller.set_l2_adapters(self._l2_adapters)\n\n        # Unified L2 eviction controller'''
    if marker not in text:
        print("❌ storage_manager.py: insertion marker not found")
        return False
    SM.write_text(text.replace(marker, insert, 1))
    compile_file(SM)
    print("✅ storage_manager.py: L2 adapters wired into L1 eviction")
    return True


def main() -> None:
    for p in [EVICT, SM, FS]:
        if not p.exists():
            print(f"❌ missing target {p}")
            sys.exit(1)
    changed = False
    changed = patch_fs_adapter() or changed
    changed = patch_eviction_controller() or changed
    changed = patch_storage_manager() or changed

    # hard verification
    if "[SYNC-L2-STORE v1]" not in FS.read_text():
        print("❌ sync store marker missing")
        sys.exit(2)
    ev = EVICT.read_text()
    if "[SYNC-L1-EVICT-FLUSH v2]" not in ev or "pop_completed_store_tasks" in ev.split("class L1EvictionController", 1)[1].split("class L2AdapterEvictionState", 1)[0]:
        print("❌ L1 eviction sync flush verification failed")
        sys.exit(3)
    print(f"\n📊 L1 eviction flush patch complete; changed={changed}")


if __name__ == "__main__":
    main()
