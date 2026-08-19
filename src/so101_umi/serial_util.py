"""List USB serial adapters (read-only)."""

from __future__ import annotations

import os
from pathlib import Path


def list_serial_devices() -> list[dict[str, str]]:
    """Return ttyACM/ttyUSB nodes plus by-id / by-path aliases."""
    found: list[dict[str, str]] = []
    nodes: list[Path] = []
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        from glob import glob

        nodes.extend(Path(p) for p in sorted(glob(pattern)))
    by_id = Path("/dev/serial/by-id")
    id_map: dict[str, str] = {}
    if by_id.is_dir():
        for link in sorted(by_id.iterdir()):
            try:
                id_map[str(link.resolve())] = str(link)
            except OSError:
                continue
    seen_real: set[str] = set()
    for node in nodes:
        try:
            real = str(node.resolve())
        except OSError:
            real = str(node)
        if real in seen_real and str(node) != real:
            continue
        seen_real.add(real)
        rec = {
            "node": str(node),
            "by_id": id_map.get(real, id_map.get(str(node), "")),
            "mode": "",
            "group": "",
        }
        try:
            st = node.stat()
            rec["mode"] = oct(st.st_mode & 0o777)
        except OSError:
            pass
        found.append(rec)
    return found


def print_serial_devices() -> None:
    rows = list_serial_devices()
    print("[serial] USB ACM/USB devices:")
    if not rows:
        print("  (none — check lsusb / dmesg; sandbox may hide /dev/tty*)")
        return
    for r in rows:
        extra = f"  by-id={r['by_id']}" if r["by_id"] else ""
        print(f"  {r['node']} mode={r['mode'] or '?'}{extra}")


def default_leader_port() -> str:
    preferred = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107971-if00"
    if os.path.exists(preferred):
        return preferred
    alt = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107039-if00"
    if os.path.exists(alt):
        return alt
    if os.path.exists("/dev/ttyACM1"):
        return "/dev/ttyACM1"
    if os.path.exists("/dev/ttyACM0"):
        return "/dev/ttyACM0"
    return preferred
