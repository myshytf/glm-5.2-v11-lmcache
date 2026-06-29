#!/usr/bin/env python3
"""
vLLM patch: accept empty `tools` array without error.

Some clients (e.g. Xcode AI assistants) send `"tools": []` in chat completion
requests. vLLM raises a ValueError for this, rejecting the request. This patch
silently removes the empty tools field instead, matching OpenAI API behavior
where omitting `tools` is equivalent to no tools.
"""
import os
import re
import sys
import traceback
from pathlib import Path

PATCH_MARKER = "EMPTY-TOOLS-PATCH"
VENV = "/opt/venv/lib/python3.12/site-packages"
TARGET = os.path.join(VENV, "vllm/entrypoints/openai/chat_completion/protocol.py")


def patch_file(path: str, old: str, new: str, check: str = None) -> bool:
    """Replace `old` with `new` in `path`. Returns True if patched."""
    if not os.path.exists(path):
        print(f"[{PATCH_MARKER}] SKIP: {path} not found")
        return False
    with open(path, "r") as f:
        content = f.read()
    if check and check in content:
        print(f"[{PATCH_MARKER}] Already patched: {os.path.basename(path)}")
        return True
    if old not in content:
        print(f"[{PATCH_MARKER}] WARN: pattern not found in {os.path.basename(path)}")
        return False
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"[{PATCH_MARKER}] Patched: {os.path.basename(path)}")
    return True


def main():
    # The original code raises ValueError when tools == []:
    #
    #   if data.get("tools") == []:
    #       raise ValueError(
    #           "`tools` must not be an empty array. "
    #           "Either provide at least one tool or omit the field entirely."
    #       )
    #
    # Replace with: silently remove the empty tools field.

    old = '''        # Reject empty tools array, matching OpenAI API behavior
        if data.get("tools") == []:
            raise ValueError(
                "`tools` must not be an empty array. "
                "Either provide at least one tool or omit the field entirely."
            )'''

    new = '''        # Accept empty tools array — silently remove it (EMPTY-TOOLS-PATCH)
        # Some clients (e.g. Xcode AI assistants) send tools=[] unconditionally.
        if data.get("tools") == []:
            data.pop("tools", None)'''

    ok = patch_file(TARGET, old, new, check="EMPTY-TOOLS-PATCH")
    if not ok:
        print(f"[{PATCH_MARKER}] Patch failed — see warnings above")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
