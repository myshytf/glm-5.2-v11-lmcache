#!/usr/bin/env python3
"""
Patch LMCache fs_l2_adapter.py to use posix_memalign-based aligned buffers
for O_DIRECT read/write.

Root cause: LMCache 0.4.6 O_DIRECT path calls os.write(fd, buf) where buf is
a Python bytes/bytearray object. Python's memory allocator does NOT guarantee
4096-byte alignment, so O_DIRECT fails with OSError: [Errno 22] Invalid argument.

This patch:
1. Adds ctypes-based posix_memalign helper for aligned buffer allocation.
2. Rewrites _write_with_odirect to copy data into an aligned buffer (with
   padding to block size) before os.write().
3. Rewrites _read_with_odirect to read into an aligned buffer, then copy
   the actual data portion to dst_buf.
4. Removes the size-alignment check (we now pad automatically).
5. Updates both async and sync store paths to skip the alignment warning
   since we handle non-aligned sizes via padding.

Works with any chunk_size (1024, 2048, etc.) — padding is automatic.
"""

import os
import re
import sys
from pathlib import Path

ADAPTER_PATH = Path(
    os.environ.get(
        "LMCACHE_FS_L2_PATH",
        "/opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py",
    )
)

BACKUP_SUFFIX = ".bak_odirect_aligned"

# ─── Helpers ──────────────────────────────────────────────────────────────────

ALIGNED_BUFFER_CODE = '''

# ── O_DIRECT aligned buffer support (patch_odirect_aligned) ──────────────
import ctypes
import ctypes.util

_libc_cdi = None


def _get_libc_cdi():
    """Return a cached handle to libc for posix_memalign/free."""
    global _libc_cdi
    if _libc_cdi is None:
        libname = ctypes.util.find_library("c") or "libc.so.6"
        _libc_cdi = ctypes.CDLL(libname)
        _libc_cdi.posix_memalign.restype = ctypes.c_int
        _libc_cdi.posix_memalign.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        _libc_cdi.free.restype = None
        _libc_cdi.free.argtypes = [ctypes.c_void_p]
    return _libc_cdi


class _AlignedBuffer:
    """Context manager: posix_memalign'd buffer that is released on exit."""

    def __init__(self, size: int, alignment: int = 4096):
        self._size = size
        self._alignment = alignment
        self._ptr = ctypes.c_void_p(0)
        self._view = None  # ctypes array view (writeable buffer protocol)

    def __enter__(self) -> "ctypes.Array":
        libc = _get_libc_cdi()
        ret = libc.posix_memalign(
            ctypes.byref(self._ptr), self._alignment, self._size
        )
        if ret != 0:
            raise MemoryError(
                f"posix_memalign(size={self._size}, align={self._alignment}) "
                f"failed: errno={ret}"
            )
        self._view = (ctypes.c_char * self._size).from_address(self._ptr.value)
        return self._view

    def __exit__(self, *exc):
        if self._ptr.value:
            _get_libc_cdi().free(self._ptr)
            self._ptr = ctypes.c_void_p(0)
        return False

# ── End O_DIRECT aligned buffer support ──────────────────────────────────
'''

NEW_WRITE_METHOD = '''    def _write_with_odirect(self, file_path: Path, buf: bytes) -> None:
        """Synchronous O_DIRECT write of *buf* using an aligned buffer.

        Data is copied into a posix_memalign'd buffer (padded to block size)
        so that both buffer-address and byte-count alignment requirements
        for O_DIRECT are satisfied.  Runs in an executor (not on the event
        loop).
        """
        fd = -1
        bs = self._os_disk_bs or 4096
        size = len(buf)
        padded = ((size + bs - 1) // bs) * bs  # round up to block boundary
        try:
            fd = os.open(
                str(file_path),
                os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_DIRECT", 0),
                0o644,
            )
            with _AlignedBuffer(padded, bs) as abuf:
                # Copy actual data.  obj.byte_array can be memoryview/bytearray,
                # not only bytes; ctypes.memmove only accepts a real ctypes
                # object/address or bytes-like object it understands, so force
                # a contiguous bytes source first.
                src = memoryview(buf)
                if src.ndim != 1 or src.format not in ("B", "b", "c"):
                    src = src.cast("B")
                src_bytes = src.tobytes()
                ctypes.memmove(ctypes.addressof(abuf), src_bytes, size)
                # Zero-fill padding (already zeroed by fresh allocation, but
                # be explicit for safety)
                if padded > size:
                    ctypes.memset(
                        ctypes.addressof(abuf) + size, 0, padded - size
                    )
                n = os.write(fd, abuf)
                if n != padded:
                    raise OSError(
                        f"Short O_DIRECT write: wrote {n}/{padded} bytes "
                        f"to {file_path}"
                    )
        except Exception:
            logger.exception("Failed to O_DIRECT write %s", file_path)
            raise
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
'''

NEW_READ_METHOD = '''    def _read_with_odirect(
        self,
        file_path: Path,
        dst_buf: Union[bytearray, memoryview, bytes],
    ) -> int:
        """Synchronous O_DIRECT read into *dst_buf* using an aligned buffer.

        Reads into a posix_memalign'd buffer (padded to block size), then
        copies the actual data portion to *dst_buf*.  Returns the number of
        real data bytes read.  Runs in an executor (not on the event loop).
        """
        fd = -1
        bs = self._os_disk_bs or 4096
        size = len(dst_buf)
        padded = ((size + bs - 1) // bs) * bs
        try:
            fd = os.open(
                str(file_path),
                os.O_RDONLY | getattr(os, "O_DIRECT", 0),
            )
            with _AlignedBuffer(padded, bs) as abuf:
                import os as _os
                with _os.fdopen(fd, "rb", buffering=0) as fdo:
                    fd = -1
                    mv = memoryview(abuf).cast("B")
                    total = 0
                    while total < padded:
                        n = fdo.readinto(mv[total:])
                        if n is None or n == 0:
                            break
                        total += n
                # Copy actual data to dst_buf
                actual = min(total, size)
                try:
                    dst_mv = dst_buf if isinstance(dst_buf, memoryview) else memoryview(dst_buf)
                    if dst_mv.readonly:
                        logger.warning("O_DIRECT destination buffer is read-only for %s", file_path)
                        return 0
                    dst_mv = dst_mv.cast("B")
                    dst_mv[:actual] = mv[:actual]
                except Exception:
                    logger.exception("Failed to copy O_DIRECT read buffer for %s", file_path)
                    return 0
                return actual
        except Exception:
            logger.exception("Failed to O_DIRECT read %s", file_path)
            return 0
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
'''


# ─── Patch logic ──────────────────────────────────────────────────────────────

def _find_method(text: str, method_name: str, end_marker: str = "\n    def ") -> tuple[int, int]:
    """Return (start, end) byte offsets of a method in *text*."""
    marker = f"    def {method_name}"
    start = text.index(marker)
    # Find the next method definition after this one
    next_def = text.index(end_marker, start + len(marker))
    return start, next_def


def apply():
    if not ADAPTER_PATH.exists():
        print(f"❌ Adapter not found: {ADAPTER_PATH}")
        sys.exit(1)

    original = ADAPTER_PATH.read_text()

    # Backup
    backup = ADAPTER_PATH.with_suffix(ADAPTER_PATH.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_text(original)
        print(f"✅ backup {backup}")
    else:
        print(f"↪️ backup already exists {backup}")

    text = original

    # 1. Insert aligned-buffer support after the imports block if needed.
    #    Re-running the script is allowed so v2 method bodies can replace v1.
    if "class _AlignedBuffer" not in text:
        insert_marker = "\n\ndef _readinto_full("
        if insert_marker not in text:
            insert_marker = "\n\ndef _async_readinto_full("
        if insert_marker not in text:
            # Fallback: insert after the first-party imports
            insert_marker = "\nfrom lmcache.logging import init_logger\n"
        text = text.replace(insert_marker, ALIGNED_BUFFER_CODE + insert_marker, 1)
        print("✅ Added _AlignedBuffer helper class")
    else:
        print("↪️ _AlignedBuffer helper already present; replacing method bodies only")

    # 2. Replace _write_with_odirect method
    w_start, w_end = _find_method(text, "_write_with_odirect")
    text = text[:w_start] + NEW_WRITE_METHOD + text[w_end:]
    print("✅ Replaced _write_with_odirect with aligned-buffer version")

    # 3. Replace _read_with_odirect method
    r_start, r_end = _find_method(text, "_read_with_odirect")
    text = text[:r_start] + NEW_READ_METHOD + text[r_end:]
    print("✅ Replaced _read_with_odirect with aligned-buffer version")

    # 4. Remove alignment warnings in async store path (_execute_store)
    #    Replace the do_odirect size check block to always proceed with O_DIRECT
    old_check = '''                try:
                    # Decide whether O_DIRECT is usable
                    do_odirect = self._use_odirect
                    if do_odirect:
                        aligned = self._os_disk_bs > 0 and size % self._os_disk_bs == 0
                        if not aligned:
                            logger.warning(
                                "Cannot use O_DIRECT for "
                                "writing size %d, not "
                                "aligned to block size "
                                "%d.",
                                size,
                                self._os_disk_bs,
                            )
                            do_odirect = False

                    if do_odirect:'''
    new_check = '''                try:
                    do_odirect = self._use_odirect  # padding handled in _write_with_odirect

                    if do_odirect:'''
    if old_check in text:
        text = text.replace(old_check, new_check, 1)
        print("✅ Removed async store alignment check (padding auto-handles)")
    else:
        print("⚠️ Async store alignment check not found (may already be simplified)")

    # 5. Remove alignment warnings in sync store path (store_objects_sync)
    old_sync_check = '''                do_odirect = self._use_odirect
                if do_odirect:
                    aligned = self._os_disk_bs > 0 and size % self._os_disk_bs == 0
                    if not aligned:
                        logger.warning(
                            "Cannot use O_DIRECT for writing size %d, not aligned to block size %d.",
                            size, self._os_disk_bs)
                        do_odirect = False'''
    new_sync_check = '''                do_odirect = self._use_odirect  # padding handled in _write_with_odirect'''
    if old_sync_check in text:
        text = text.replace(old_sync_check, new_sync_check, 1)
        print("✅ Removed sync store alignment check (padding auto-handles)")
    else:
        print("⚠️ Sync store alignment check not found (may already be simplified)")

    # 6. Remove alignment warning in _read_with_odirect old path
    #    (already replaced the whole method, so this is a no-op)

    # 7. Syntax check
    ADAPTER_PATH.write_text(text)
    compile(text, str(ADAPTER_PATH), "exec")
    print("✅ Syntax check: OK")

    # 8. Verify key markers
    assert "_AlignedBuffer" in text, "_AlignedBuffer not found after patch"
    assert "posix_memalign" in text, "posix_memalign not found after patch"
    assert "ctypes.memmove" in text, "ctypes.memmove not found after patch"
    print("✅ Verification: all markers present")

    print("\n📊 patch_odirect_aligned complete; changed=True")


if __name__ == "__main__":
    apply()
