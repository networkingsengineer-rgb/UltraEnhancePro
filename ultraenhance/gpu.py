"""GPU / VRAM detection and live load readings."""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import List, Optional

import psutil

from . import engines
from .config import log


@dataclass
class GPUInfo:
    index: int          # Vulkan device index used by the ncnn engines (-g)
    name: str
    vram_mb: int        # 0 = unknown
    vendor: str         # NVIDIA / AMD / Intel / Other

    @property
    def label(self) -> str:
        v = f"{self.vram_mb / 1024:.1f} GB" if self.vram_mb else "VRAM unknown"
        return f"[{self.index}] {self.name} ({v})"


def _vendor(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n or "rtx" in n:
        return "NVIDIA"
    if "amd" in n or "radeon" in n:
        return "AMD"
    if "intel" in n or "arc" in n or "iris" in n or "uhd" in n:
        return "Intel"
    return "Other"


def _windows_adapters() -> List[tuple[str, int]]:
    """Name + VRAM from the registry (exact VRAM, not the 4 GB-capped WMI value)."""
    out = []
    if os.name != "nt":
        return out
    try:
        import winreg
        root = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root) as k:
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(k, i)
                except OSError:
                    break
                i += 1
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(k, sub) as sk:
                        name = winreg.QueryValueEx(sk, "DriverDesc")[0]
                        vram = 0
                        for val in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                            try:
                                raw = winreg.QueryValueEx(sk, val)[0]
                                if isinstance(raw, bytes):
                                    raw = int.from_bytes(raw[:8], "little")
                                vram = int(raw) // (1024 * 1024)
                                if vram:
                                    break
                            except OSError:
                                continue
                        if "basic" not in name.lower() and "virtual" not in name.lower():
                            out.append((name, vram))
                except OSError:
                    continue
    except Exception as e:  # pragma: no cover - Windows only
        log.debug("registry GPU scan failed: %s", e)
    return out


def _vulkan_devices() -> List[str]:
    """Ask Real-ESRGAN which Vulkan devices it sees (its order == -g index)."""
    try:
        exe = engines.realesrgan_path()
    except engines.EngineMissing:
        return []
    try:
        # Running with no input prints the device list to stderr and exits.
        res = engines.run([str(exe), "-i", "__none__.png", "-o", "__none__.png"], timeout=20)
        text = (res.stdout or "") + (res.stderr or "")
        names = re.findall(r"\[(\d+)\s+([^\]]+)\]\s+queueC", text)
        uniq = {}
        for idx, name in names:
            uniq.setdefault(int(idx), name.strip())
        return [uniq[i] for i in sorted(uniq)]
    except Exception as e:
        log.debug("vulkan device probe failed: %s", e)
        return []


def detect_gpus() -> List[GPUInfo]:
    vk = _vulkan_devices()
    adapters = _windows_adapters()
    nvsmi = _nvidia_smi_memory()
    gpus: List[GPUInfo] = []
    names = vk or [a[0] for a in adapters]
    for i, name in enumerate(names):
        vram = 0
        for aname, avram in adapters:
            if aname.lower() in name.lower() or name.lower() in aname.lower():
                vram = avram
                break
        if not vram and _vendor(name) == "NVIDIA" and nvsmi:
            vram = nvsmi.get(name, next(iter(nvsmi.values())))
        gpus.append(GPUInfo(i, name, vram, _vendor(name)))
    return gpus


def _nvidia_smi_memory() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}
    try:
        r = engines.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], timeout=5)
        res = {}
        for line in r.stdout.strip().splitlines():
            name, mem = [x.strip() for x in line.split(",")]
            res[name] = int(float(mem))
        return res
    except Exception:
        return {}


def gpu_load_percent() -> Optional[int]:
    """GPU utilisation. NVIDIA via nvidia-smi; others via Windows perf counters."""
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            r = engines.run([exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], timeout=3)
            return int(r.stdout.strip().splitlines()[0])
        except Exception:
            pass
    if os.name == "nt":
        try:
            ps = ("(Get-Counter '\\GPU Engine(*engtype_3D)\\Utilization Percentage' -ErrorAction SilentlyContinue)"
                  ".CounterSamples | Measure-Object -Property CookedValue -Sum | Select -Expand Sum")
            r = engines.run(["powershell", "-NoProfile", "-Command", ps], timeout=6)
            return min(100, int(float(r.stdout.strip() or 0)))
        except Exception:
            return None
    return None


def cpu_load_percent() -> int:
    return int(psutil.cpu_percent(interval=None))


def pick_gpu(gpus: List[GPUInfo], wanted: int) -> Optional[GPUInfo]:
    """wanted: -2 auto (largest VRAM, prefer discrete), -1 CPU, else index."""
    if wanted == -1 or not gpus:
        return None
    if wanted >= 0:
        for g in gpus:
            if g.index == wanted:
                return g
    return sorted(gpus, key=lambda g: (g.vendor != "Intel", g.vram_mb), reverse=True)[0]
