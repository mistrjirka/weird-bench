#!/usr/bin/env python3
"""
Utility functions to list Vulkan GPUs using text output from `vulkaninfo`.

Design goals:
- Do NOT use `vulkaninfo --json` (often broken/inconsistent across distros)
- Parse lines like: "GPU id = 0 (Device Name ...)" from plain `vulkaninfo`
- Extract outermost parentheses content as name (handles nested parentheses)
- Ignore software renderers such as llvmpipe/lavapipe/swiftshader/swrast
- Return indices compatible with GGML_VK_VISIBLE_DEVICES
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional


SOFTWARE_RENDERER_KEYWORDS = [
    "llvmpipe", "lavapipe", "swiftshader", "swrast", "software", "mesa software",
]


def _detect_driver_from_name(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["nvidia", "geforce", "gtx", "rtx", "quadro", "tesla"]):
        return "nvidia"
    if any(k in n for k in ["amd", "radeon", "rx ", "vega", "fury"]):
        return "amd"
    if any(k in n for k in ["intel", "iris", "uhd", "arc"]):
        return "intel"
    return "vulkan"


def get_vulkaninfo_text(timeout: int = 30) -> Optional[str]:
    """Run `vulkaninfo` and return stdout as text; suppress stderr noise.

    Returns None if executable not found or command fails/produces no output.
    """
    if not shutil.which("vulkaninfo"):
        return None
    try:
        r = subprocess.run(
            ["vulkaninfo"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # match user's awk approach (2>/dev/null)
            text=True,
            timeout=timeout,
        )
        out = (r.stdout or "").strip()
        return out or None
    except Exception:
        return None


def parse_vulkaninfo_text(text: str) -> List[Dict[str, Any]]:
    """Parse plain text from `vulkaninfo` and return GPU list.

    Extract lines like: 'GPU id = 0 (Device Name)' -> { index, name, driver }
    Ignores software/pipe renderers.
    """
    devices: List[Dict[str, Any]] = []
    seen = set()

    for line in text.splitlines():
        # Match 'GPU id = <num>' pattern
        m = re.search(r"GPU id\s*=\s*([0-9]+)", line)
        if not m:
            continue
        idx = int(m.group(1))

        # Extract outermost parentheses content as device name
        p1 = line.find("(")
        p2 = line.rfind(")")
        name = f"Vulkan Device {idx}"
        if p1 >= 0 and p2 > p1:
            candidate = line[p1 + 1 : p2].strip()
            if candidate:
                name = candidate

        lname = name.lower()
        if any(k in lname for k in SOFTWARE_RENDERER_KEYWORDS):
            # Skip llvmpipe/lavapipe/etc.
            continue

        key = (idx, name)
        if key in seen:
            continue
        seen.add(key)

        devices.append({
            "index": idx,
            "name": name,
            "driver": _detect_driver_from_name(name),
            "icd_path": None,
        })

    return devices


def list_vulkan_gpus() -> List[Dict[str, Any]]:
    """High-level helper: run vulkaninfo (text) and parse devices.

    Returns empty list if nothing is found. Never raises.
    """
    text = get_vulkaninfo_text()
    if not text:
        return []
    return parse_vulkaninfo_text(text)
