#!/usr/bin/env python3
"""
Patch LMCache 0.4.6 fs_l2_adapter.py — 4 critical fixes:

1. FSL2AdapterConfig: add max_capacity_gb (was silently ignored)
2. FSL2Adapter: pass max_capacity_bytes to base class → enables L2 eviction
3. _execute_store: call _notify_keys_stored → fixes l2_usage_bytes = 0
4. delete(): implement actual file deletion + _notify_keys_deleted
5. Startup scan: count existing files for accurate usage on boot

Without this patch:
  - L2 disk grows unbounded (eviction controller skips fs adapter)
  - l2_usage_bytes metric always reports 0
  - L2 eviction controller.adapters is always empty
"""

import sys
from pathlib import Path

FILE = Path(
    "/opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/"
    "l2_adapters/fs_l2_adapter.py"
)


def patch(text: str) -> tuple[str, list[str]]:
    changes = []
    orig = text

    # ── 1. FSL2AdapterConfig.__init__: add max_capacity_gb ──────────────
    old = '''\
    def __init__(
        self,
        base_path: str,
        relative_tmp_dir: Optional[str] = None,
        read_ahead_size: Optional[int] = None,
        use_odirect: bool = False,
    ):
        """Initialize FSL2AdapterConfig.

        Args:
            base_path: Directory for storing KV cache files.
            relative_tmp_dir: Relative sub-dir under
                base_path for temp files during writes.
            read_ahead_size: If set, trigger filesystem
                readahead by issuing a small initial read
                of this many bytes before reading the rest.
            use_odirect: If True, bypass the OS page cache
                using O_DIRECT for both reads and writes.
                Requires buffer sizes aligned to the
                filesystem block size.
        """
        self.base_path = base_path
        self.relative_tmp_dir = relative_tmp_dir
        self.read_ahead_size = read_ahead_size
        self.use_odirect = use_odirect'''

    new = '''\
    def __init__(
        self,
        base_path: str,
        relative_tmp_dir: Optional[str] = None,
        read_ahead_size: Optional[int] = None,
        use_odirect: bool = False,
        max_capacity_gb: float = 0,
    ):
        """Initialize FSL2AdapterConfig.

        Args:
            base_path: Directory for storing KV cache files.
            relative_tmp_dir: Relative sub-dir under
                base_path for temp files during writes.
            read_ahead_size: If set, trigger filesystem
                readahead by issuing a small initial read
                of this many bytes before reading the rest.
            use_odirect: If True, bypass the OS page cache
                using O_DIRECT for both reads and writes.
                Requires buffer sizes aligned to the
                filesystem block size.
            max_capacity_gb: Maximum L2 capacity in GB.
                0 means unlimited / no eviction (default).
                When > 0, the eviction controller keeps
                disk usage below this limit.
        """
        self.base_path = base_path
        self.relative_tmp_dir = relative_tmp_dir
        self.read_ahead_size = read_ahead_size
        self.use_odirect = use_odirect
        self.max_capacity_gb = max_capacity_gb'''

    if old in text:
        text = text.replace(old, new)
        changes.append("FSL2AdapterConfig.__init__: added max_capacity_gb")
    else:
        print("WARN: could not patch FSL2AdapterConfig.__init__", file=sys.stderr)

    # ── 2. FSL2AdapterConfig.from_dict: parse max_capacity_gb ───────────
    old_from_dict = '''\
        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")
        return cls(
            base_path=base_path,
            relative_tmp_dir=relative_tmp_dir,
            read_ahead_size=read_ahead_size,
            use_odirect=use_odirect,
        )'''

    new_from_dict = '''\
        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")
        max_capacity_gb = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_gb, (int, float)) or max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be a non-negative number")
        return cls(
            base_path=base_path,
            relative_tmp_dir=relative_tmp_dir,
            read_ahead_size=read_ahead_size,
            use_odirect=use_odirect,
            max_capacity_gb=float(max_capacity_gb),
        )'''

    if old_from_dict in text:
        text = text.replace(old_from_dict, new_from_dict)
        changes.append("FSL2AdapterConfig.from_dict: parse max_capacity_gb")
    else:
        print("WARN: could not patch FSL2AdapterConfig.from_dict", file=sys.stderr)

    # ── 3. FSL2AdapterConfig.help: document max_capacity_gb ─────────────
    old_help = '''\
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (optional, default false)"\
'''

    new_help = '''\
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (optional, default false)\\n"
            "- max_capacity_gb (float): max L2 capacity "
            "in GB; 0 = unlimited / no eviction (default 0)"\
'''

    if old_help in text:
        text = text.replace(old_help, new_help)
        changes.append("FSL2AdapterConfig.help: documented max_capacity_gb")
    else:
        print("WARN: could not patch FSL2AdapterConfig.help", file=sys.stderr)

    # ── 4. FSL2Adapter.__init__: pass max_capacity_bytes to base ────────
    old_init = '''\
    def __init__(self, config: FSL2AdapterConfig):
        super().__init__()
        self._config = config'''

    new_init = '''\
    def __init__(self, config: FSL2AdapterConfig):
        max_cap_bytes = (
            int(config.max_capacity_gb * 1e9)
            if config.max_capacity_gb > 0
            else 0
        )
        super().__init__(max_capacity_bytes=max_cap_bytes)
        self._config = config'''

    if old_init in text:
        text = text.replace(old_init, new_init)
        changes.append("FSL2Adapter.__init__: pass max_capacity_bytes to base")
    else:
        print("WARN: could not patch FSL2Adapter.__init__", file=sys.stderr)

    # ── 5. Add _scan_existing_files after _loop_thread start ───────────
    # Insert after the event loop thread start, before logger.info
    old_thread = '''\
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        logger.info(
            "Initialized FSL2Adapter with base_path=%s, "'''

    new_thread = '''\
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        # Scan existing files for accurate byte accounting on startup.
        # Without this, _total_bytes_used stays 0 after restart, making
        # the eviction controller blind to pre-existing disk usage.
        self._scan_existing_files()

        logger.info(
            "Initialized FSL2Adapter with base_path=%s, "'''

    if old_thread in text:
        text = text.replace(old_thread, new_thread)
        changes.append("FSL2Adapter.__init__: added _scan_existing_files call")
    else:
        print("WARN: could not patch _scan_existing_files call", file=sys.stderr)

    # ── 6. Add _scan_existing_files method + register_listener hook ─────
    old_run_loop = '''\
    def _run_event_loop(self) -> None:'''

    new_run_loop = '''\
    _scanned_keys: list[ObjectKey] = []
    _scanned_sizes: list[int] = []
    _scan_done: bool = False

    def _scan_existing_files(self) -> None:
        """Walk base_path and enumerate existing .data files.

        Stores the keys/sizes for later replay when listeners are
        registered (which happens after __init__ in StorageManager).
        Also updates _total_bytes_used / _bytes_by_cache_salt via
        _notify_keys_stored so the l2_usage_bytes gauge works.
        """
        try:
            for f in self._base_path.glob("*.data"):
                try:
                    size = f.stat().st_size
                    key = _filename_to_object_key(f.name)
                    if key is not None:
                        self._scanned_keys.append(key)
                        self._scanned_sizes.append(size)
                except OSError:
                    pass
        except Exception:
            logger.exception("FSL2Adapter startup scan failed")
            return
        # Update byte accounting immediately (gauge works even
        # before listeners are registered).
        if self._scanned_keys:
            with self._usage_lock:
                total = sum(self._scanned_sizes)
                self._total_bytes_used = total
                by_salt: dict[str, int] = {}
                for k, s in zip(self._scanned_keys, self._scanned_sizes):
                    by_salt[k.cache_salt] = by_salt.get(k.cache_salt, 0) + s
                self._bytes_by_cache_salt = by_salt
            logger.info(
                "FSL2Adapter startup scan: %.2f GB in %d existing files "
                "(max_capacity_gb=%.1f)",
                total / 1e9,
                len(self._scanned_keys),
                self._config.max_capacity_gb,
            )
        self._scan_done = True

    def register_listener(self, listener) -> None:
        """Override to replay scanned keys to newly-registered listeners.

        StorageManager registers L2EvictionPolicy listeners AFTER
        FSL2Adapter.__init__, so a startup scan during __init__ would
        fire _notify_keys_stored before any listener exists.  We
        replay the scanned keys to each new listener so the eviction
        controller's LRU list reflects pre-existing files.
        """
        super().register_listener(listener)
        if self._scan_done and self._scanned_keys:
            listener.on_l2_keys_stored(self._scanned_keys)

    def _run_event_loop(self) -> None:'''

    if old_run_loop in text:
        text = text.replace(old_run_loop, new_run_loop)
        changes.append("Added _scan_existing_files + register_listener hook")
    else:
        print("WARN: could not add _scan_existing_files method", file=sys.stderr)

    # ── 7. _execute_store: add _notify_keys_stored ─────────────────────
    old_store = '''\
    async def _execute_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        success = True
        bytes_written = 0
        try:
            for key, obj in zip(keys, objects, strict=True):
                file_path, tmp_path = self._key_to_file_and_tmp_path(key)

                # Skip if already stored on disk
                if await aiofiles.os.path.exists(file_path):
                    continue'''

    new_store = '''\
    async def _execute_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        success = True
        bytes_written = 0
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        try:
            for key, obj in zip(keys, objects, strict=True):
                file_path, tmp_path = self._key_to_file_and_tmp_path(key)

                # Skip if already stored on disk
                if await aiofiles.os.path.exists(file_path):
                    continue'''

    if old_store in text:
        text = text.replace(old_store, new_store)
        changes.append("_execute_store: added stored_keys/stored_sizes tracking")
    else:
        print("WARN: could not patch _execute_store init", file=sys.stderr)

    # ── 8. _execute_store: append to stored_keys on successful write ────
    # After "bytes_written += size" and logger.debug, add tracking
    old_debug = '''\
                    await aiofiles.os.replace(tmp_path, file_path)
                    bytes_written += size
                    logger.debug(
                        "FSL2Adapter stored key %s (%d bytes)",
                        file_path.name,
                        size,
                    )'''

    new_debug = '''\
                    await aiofiles.os.replace(tmp_path, file_path)
                    bytes_written += size
                    stored_keys.append(key)
                    stored_sizes.append(size)
                    logger.debug(
                        "FSL2Adapter stored key %s (%d bytes)",
                        file_path.name,
                        size,
                    )'''

    if old_debug in text:
        text = text.replace(old_debug, new_debug)
        changes.append("_execute_store: track stored keys/sizes on write")
    else:
        print("WARN: could not patch _execute_store debug block", file=sys.stderr)

    # ── 9. _execute_store: call _notify_keys_stored before completion ───
    old_complete = '''\
        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(success, bytes_written)
        self._store_efd.notify()'''

    new_complete = '''\
        # Notify base class byte accounting so l2_usage_bytes gauge and
        # eviction controller see the new data.
        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)

        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(success, bytes_written)
        self._store_efd.notify()'''

    if old_complete in text:
        text = text.replace(old_complete, new_complete)
        changes.append("_execute_store: call _notify_keys_stored on completion")
    else:
        print("WARN: could not patch _execute_store completion", file=sys.stderr)

    # ── 10. Replace no-op delete() with real implementation ─────────────
    old_delete = '''\
    def delete(self, keys: list[ObjectKey]) -> None:
        # Not implemented for the filesystem adapter.
        pass

    # ``get_usage()`` is inherited from ``L2AdapterInterface``. The FS
    # adapter declares no max capacity (default 0) so ``supports_global_eviction``
    # returns ``False`` and ``usage_fraction == -1.0`` — the eviction
    # controller treats this as "no eviction signal" and skips the
    # adapter entirely.'''

    new_delete = '''\
    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete files from disk and update byte accounting.

        Called by the L2 eviction controller when disk usage exceeds
        the configured watermark.
        """
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        for key in keys:
            file_path = self._key_to_path(key)
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                deleted_keys.append(key)
                deleted_sizes.append(size)
                logger.debug(
                    "FSL2Adapter evicted key %s (%d bytes)",
                    file_path.name,
                    size,
                )
            except FileNotFoundError:
                pass
            except Exception:
                logger.exception(
                    "FSL2Adapter failed to delete %s",
                    file_path,
                )
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    # ``get_usage()`` is inherited from ``L2AdapterInterface``.  When
    # ``max_capacity_gb > 0`` is set, ``supports_global_eviction``
    # returns ``True`` and ``usage_fraction`` reflects actual disk
    # usage — the eviction controller will trigger LRU eviction when
    # usage exceeds the configured watermark.'''

    if old_delete in text:
        text = text.replace(old_delete, new_delete)
        changes.append("delete(): implemented file deletion + _notify_keys_deleted")
    else:
        print("WARN: could not patch delete()", file=sys.stderr)

    # ── 11. Update logger.info to include max_capacity_gb ───────────────
    old_log = '''\
        logger.info(
            "Initialized FSL2Adapter with base_path=%s, "
            "relative_tmp_dir=%s, "
            "read_ahead_size=%s, use_odirect=%s",
            self._base_path,
            self._relative_tmp_dir,
            self._read_ahead_size,
            self._use_odirect,
        )'''

    new_log = '''\
        logger.info(
            "Initialized FSL2Adapter with base_path=%s, "
            "relative_tmp_dir=%s, "
            "read_ahead_size=%s, use_odirect=%s, "
            "max_capacity_gb=%.1f",
            self._base_path,
            self._relative_tmp_dir,
            self._read_ahead_size,
            self._use_odirect,
            self._config.max_capacity_gb,
        )'''

    if old_log in text:
        text = text.replace(old_log, new_log)
        changes.append("logger.info: added max_capacity_gb")
    else:
        print("WARN: could not patch logger.info", file=sys.stderr)

    return text, changes


def main():
    if not FILE.exists():
        print(f"ERROR: {FILE} not found", file=sys.stderr)
        sys.exit(1)

    # Check if already patched (look for the register_listener override)
    text = FILE.read_text()
    if "register_listener" in text and "_scanned_keys" in text:
        print("Already patched — skipping.")
        return

    patched, changes = patch(text)

    if not changes:
        print("ERROR: no patches applied — string matching failed", file=sys.stderr)
        sys.exit(1)

    # Write patched file
    FILE.write_text(patched)
    print(f"Patched {FILE} — {len(changes)} changes:")
    for c in changes:
        print(f"  ✓ {c}")

    # Verify syntax
    import py_compile
    try:
        py_compile.compile(str(FILE), doraise=True)
        print("Syntax check: OK")
    except py_compile.PyCompileError as e:
        print(f"SYNTAX ERROR after patching — reverting!", file=sys.stderr)
        FILE.write_text(text)
        sys.exit(1)


if __name__ == "__main__":
    main()
