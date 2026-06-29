"""
Patch: L1EvictionController — flush TTL-expired objects to L2 even when
memory is below the watermark.

Without this patch, objects whose read_lock TTL has expired remain in L1
but are never flushed to L2 unless memory pressure triggers eviction.
This means popular chunks that expire by TTL are lost from both L1 and L2
when memory is not under pressure.

The patch adds a secondary pass in the eviction loop: when memory is below
the watermark, it still scans for TTL-expired (evictable) objects and flushes
them to L2. This ensures L2 always has a copy before the object is eventually
removed from L1.

Applied to: lmcache/v1/distributed/storage_controllers/eviction_controller.py
"""

import re
import sys

PATCH_MARKER = "# [PATCH: TTL-FLUSH]"

def apply_patch(content: str) -> str:
    if PATCH_MARKER in content:
        print("Already patched, skipping")
        return content

    # 1. Patch eviction_loop: add TTL-expired flush pass when below watermark
    old_loop = '''    def eviction_loop(self):
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
            self._publish_triggered(usage, watermark)'''

    new_loop = '''    def eviction_loop(self):
        watermark = self._eviction_config.trigger_watermark
        eviction_ratio = self._eviction_config.eviction_ratio

        while not self._stop_flag.is_set():
            time.sleep(1)
            used_bytes, total_bytes = self._l1_manager.get_memory_usage()
            usage = 0 if total_bytes == 0 else used_bytes / total_bytes
            if usage < watermark:
                # [PATCH: TTL-FLUSH] Even below watermark, flush TTL-expired
                # objects to L2 so they persist before being lost from L1.
                if self._l2_adapters:
                    self._flush_expired_to_l2()
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

    def _flush_expired_to_l2(self):
        """[PATCH: TTL-FLUSH] Flush evictable (TTL-expired) objects to L2.

        Scans L1 for objects whose read_lock TTL has expired (making them
        evictable) and flushes them to L2 before they become stale. This
        runs even when memory is below the watermark, ensuring L2 always
        has a copy of expired L1 objects.
        """
        if not self._l2_adapters:
            return
        # Get all evictable keys (TTL-expired, not locked)
        evictable_keys = []
        for key, entry in self._l1_manager._objects.items():
            if self._l1_manager.is_key_evictable(key):
                evictable_keys.append(key)
        if not evictable_keys:
            return
        # Limit batch size to avoid blocking the eviction loop
        batch = evictable_keys[:64]
        logger.info(
            "TTL-FLUSH: flushing %d/%d expired keys to L2 (memory below watermark)",
            len(batch), len(evictable_keys))
        self._flush_to_l2_then_delete(batch)'''

    if old_loop not in content:
        print("ERROR: Could not find eviction_loop to patch")
        sys.exit(1)

    content = content.replace(old_loop, new_loop)
    print("Patch applied successfully")
    return content


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/storage_controllers/eviction_controller.py"
    with open(path, "r") as f:
        content = f.read()
    patched = apply_patch(content)
    with open(path, "w") as f:
        f.write(patched)
    print(f"Written to {path}")
