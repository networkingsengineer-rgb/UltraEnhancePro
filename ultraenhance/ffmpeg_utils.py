"""FFmpeg probing, encoder detection and command building.

All command builders are pure functions (they only return lists) so they
can be unit-tested without FFmpeg installed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

from .presets import QUALITY_PRESETS

# Subtitle codecs that are text (can become mov_text in MP4).
TEXT_SUBS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
# Audio codecs MP4 can carry as-is.
MP4_AUDIO_OK = {"aac", "mp3", "ac3", "eac3", "alac", "opus", "flac"}
WEBM_AUDIO_OK = {"opus", "vorbis"}

CODECS = ("H.264", "H.265/HEVC", "AV1")

# encoder family -> codec -> ffmpeg encoder name
ENCODERS: Dict[str, Dict[str, str]] = {
    "nvenc": {"H.264": "h264_nvenc", "H.265/HEVC": "hevc_nvenc", "AV1": "av1_nvenc"},
    "amf": {"H.264": "h264_amf", "H.265/HEVC": "hevc_amf", "AV1": "av1_amf"},
    "qsv": {"H.264": "h264_qsv", "H.265/HEVC": "hevc_qsv", "AV1": "av1_qsv"},
    "software": {"H.264": "libx264", "H.265/HEVC": "libx265", "AV1": "libsvtav1"},
}


@dataclass
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str


@dataclass
class MediaInfo:
    width: int
    height: int
    fps: Fraction
    duration: float
    frames: int
    interlaced: bool = False
    has_audio: bool = False
    streams: List[StreamInfo] = field(default_factory=list)
    pix_fmt: str = ""

    @property
    def fps_float(self) -> float:
        return float(self.fps) if self.fps else 0.0

    def audio_codecs(self) -> List[str]:
        return [s.codec_name for s in self.streams if s.codec_type == "audio"]

    def subtitle_streams(self) -> List[StreamInfo]:
        return [s for s in self.streams if s.codec_type == "subtitle"]


def parse_fps(text: str) -> Fraction:
    try:
        f = Fraction(text)
        return f if f > 0 else Fraction(0)
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def probe_cmd(ffprobe: str, path: str) -> List[str]:
    return [ffprobe, "-v", "error", "-print_format", "json", "-show_format",
            "-show_streams", str(path)]


def parse_probe(data: dict) -> MediaInfo:
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    if v is None:
        raise ValueError("This file has no video stream.")
    fps = parse_fps(v.get("avg_frame_rate", "0/0")) or parse_fps(v.get("r_frame_rate", "0/0"))
    if not fps:
        fps = Fraction(30)
    dur = float(v.get("duration") or data.get("format", {}).get("duration") or 0)
    nb = int(v.get("nb_frames") or 0)
    if not nb and dur:
        nb = int(round(dur * float(fps)))
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    # Respect rotation metadata (phones record portrait as rotated landscape).
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = abs(int(sd["rotation"]))
    if "rotate" in v.get("tags", {}):
        rot = abs(int(v["tags"]["rotate"]))
    if rot in (90, 270):
        w, h = h, w
    field_order = v.get("field_order", "progressive")
    return MediaInfo(
        width=w, height=h, fps=fps, duration=dur, frames=nb,
        interlaced=field_order not in ("progressive", "unknown", ""),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        streams=[StreamInfo(int(s.get("index", i)), s.get("codec_type", ""), s.get("codec_name", ""))
                 for i, s in enumerate(streams)],
        pix_fmt=v.get("pix_fmt", ""),
    )


def parse_encoders(text: str) -> set:
    names = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return names


def encoder_test_cmd(ffmpeg: str, encoder: str) -> List[str]:
    """Tiny real encode: proves the hardware/driver works, not just that ffmpeg lists it."""
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "color=black:s=256x256:d=0.1", "-frames:v", "2", "-c:v", encoder, "-f", "null", "-"]


def choose_encoder(codec: str, available: set, family: str = "auto") -> str:
    """Pick the best working encoder for a codec.

    available = encoders that passed encoder_test_cmd. family = 'auto' or a key of ENCODERS.
    """
    if codec not in CODECS:
        raise ValueError(f"Unknown codec {codec}")
    order = ["nvenc", "amf", "qsv", "software"] if family == "auto" else [family, "software"]
    for fam in order:
        enc = ENCODERS[fam][codec]
        if enc in available:
            return enc
    # Last-chance software fallbacks
    for enc in {"AV1": ["libsvtav1", "libaom-av1"], "H.265/HEVC": ["libx265"], "H.264": ["libx264"]}[codec]:
        if enc in available:
            return enc
    return "libx264"


def encoder_quality_args(encoder: str, quality: str = "Balanced",
                         crf: Optional[int] = None, bitrate_kbps: Optional[int] = None) -> List[str]:
    sw_preset, default_crf, hw_preset = QUALITY_PRESETS[quality]
    q = crf if crf is not None else default_crf
    if bitrate_kbps:
        rate = [f"-b:v", f"{bitrate_kbps}k", "-maxrate", f"{int(bitrate_kbps * 1.5)}k",
                "-bufsize", f"{bitrate_kbps * 2}k"]
    else:
        rate = None
    if encoder in ("libx264", "libx265"):
        args = ["-preset", sw_preset]
        args += rate if rate else ["-crf", str(q)]
        if encoder == "libx265":
            args += ["-tag:v", "hvc1", "-x265-params", "log-level=error"]
        return args
    if encoder == "libsvtav1":
        svt = {"Fast": "10", "Balanced": "7", "Best": "4"}[quality]
        return ["-preset", svt] + (rate if rate else ["-crf", str(q + 8)])
    if encoder == "libaom-av1":
        cpu = {"Fast": "8", "Balanced": "6", "Best": "4"}[quality]
        return ["-cpu-used", cpu, "-row-mt", "1"] + (rate if rate else ["-crf", str(q + 10), "-b:v", "0"])
    if encoder.endswith("_nvenc"):
        args = ["-preset", hw_preset, "-tune", "hq"]
        args += rate if rate else ["-rc", "vbr", "-cq", str(q + 2), "-b:v", "0"]
        if encoder.startswith("hevc"):
            args += ["-tag:v", "hvc1"]
        return args
    if encoder.endswith("_amf"):
        amf_q = {"Fast": "speed", "Balanced": "balanced", "Best": "quality"}[quality]
        args = ["-quality", amf_q]
        args += rate if rate else ["-rc", "cqp", "-qp_i", str(q), "-qp_p", str(q + 2)]
        if encoder.startswith("h264"):
            args += []
        elif encoder.startswith("hevc"):
            args += ["-tag:v", "hvc1"]
        return args
    if encoder.endswith("_qsv"):
        qsv_p = {"Fast": "veryfast", "Balanced": "medium", "Best": "veryslow"}[quality]
        args = ["-preset", qsv_p]
        args += rate if rate else ["-global_quality", str(q + 2)]
        if encoder.startswith("hevc"):
            args += ["-tag:v", "hvc1"]
        return args
    return rate if rate else []


def pre_filter_chain(deinterlace: bool = False, denoise: bool = False,
                     stabilize_trf: Optional[str] = None) -> List[str]:
    filters = []
    if deinterlace:
        filters.append("bwdif=mode=send_frame:parity=auto:deint=all")
    if stabilize_trf:
        trf = stabilize_trf.replace("\\", "/").replace(":", "\\:")
        filters.append(f"vidstabtransform=input='{trf}':smoothing=15:zoom=2,unsharp=5:5:0.6:3:3:0.3")
    if denoise:
        filters.append("hqdn3d=3:2:4:3")
    return filters


def vidstab_detect_cmd(ffmpeg: str, src: str, trf: str, start: Optional[float] = None,
                       duration: Optional[float] = None) -> List[str]:
    trf_esc = trf.replace("\\", "/").replace(":", "\\:")
    cmd = [ffmpeg, "-hide_banner", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", src]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vf", f"vidstabdetect=shakiness=6:accuracy=12:result='{trf_esc}'", "-f", "null", "-"]
    return cmd


def extract_frames_cmd(ffmpeg: str, src: str, out_pattern: str, fps: Fraction,
                       filters: Optional[List[str]] = None, start: Optional[float] = None,
                       duration: Optional[float] = None, start_number: int = 1) -> List[str]:
    """Decode video to numbered PNGs at a constant frame rate (keeps A/V sync)."""
    cmd = [ffmpeg, "-hide_banner", "-y", "-loglevel", "error", "-stats"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", src]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    vf = list(filters or []) + [f"fps={fps.numerator}/{fps.denominator}", "format=rgb24"]
    cmd += ["-map", "0:v:0", "-vf", ",".join(vf), "-fps_mode", "cfr",
            "-start_number", str(start_number), out_pattern]
    return cmd


def scale_filter(canvas_w: int, canvas_h: int, scaled_w: int, scaled_h: int,
                 offset_x: int, offset_y: int) -> str:
    """Final resize to the exact preset size (fit/fill/pad) with Lanczos."""
    f = f"scale={scaled_w}:{scaled_h}:flags=lanczos"
    if offset_x < 0 or offset_y < 0:  # fill -> crop
        f += f",crop={canvas_w}:{canvas_h}:{-min(0, offset_x)}:{-min(0, offset_y)}"
    elif offset_x > 0 or offset_y > 0 or (canvas_w, canvas_h) != (scaled_w, scaled_h):
        f += f",pad={canvas_w}:{canvas_h}:{offset_x}:{offset_y}:black"
    return f + ",setsar=1"


def _pix_fmt_for(encoder: str) -> str:
    if encoder.endswith("_qsv") or encoder.endswith("_amf"):
        return "nv12"
    return "yuv420p"


def segment_encode_cmd(ffmpeg: str, frames_pattern: str, fps: Fraction, out: str, encoder: str,
                       scale_vf: str, quality: str = "Balanced", crf: Optional[int] = None,
                       bitrate_kbps: Optional[int] = None, start_number: int = 1) -> List[str]:
    """Encode one chunk of upscaled PNG frames into a video-only segment (.mkv)."""
    cmd = [ffmpeg, "-hide_banner", "-y", "-loglevel", "error", "-stats",
           "-framerate", f"{fps.numerator}/{fps.denominator}",
           "-start_number", str(start_number), "-i", frames_pattern,
           "-map", "0:v:0", "-vf", scale_vf, "-c:v", encoder]
    cmd += encoder_quality_args(encoder, quality, crf, bitrate_kbps)
    cmd += ["-pix_fmt", _pix_fmt_for(encoder), "-r", f"{fps.numerator}/{fps.denominator}", out]
    return cmd


def concat_list_text(segments: List[str]) -> str:
    lines = []
    for s in segments:
        p = str(s).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{p}'")
    return "\n".join(lines) + "\n"


def mux_cmd(ffmpeg: str, concat_list: str, out: str, source: Optional[str] = None,
            info: Optional[MediaInfo] = None, container: str = ".mp4",
            start: Optional[float] = None, duration: Optional[float] = None) -> List[str]:
    """Join the encoded segments (no re-encode) and copy audio + subtitles from the source.

    The frames follow the source timeline at a constant rate, so the joined
    video has the same duration as the source and audio stays in sync.
    """
    container = container.lower()
    cmd = [ffmpeg, "-hide_banner", "-y", "-loglevel", "error", "-stats",
           "-f", "concat", "-safe", "0", "-i", concat_list]
    if source:
        if start is not None:
            cmd += ["-ss", f"{start:.3f}"]
        if duration is not None:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-i", source]
    cmd += ["-map", "0:v:0", "-c:v", "copy"]
    if source and info is not None:
        acodecs = info.audio_codecs()
        if acodecs:
            cmd += ["-map", "1:a?"]
            ok = MP4_AUDIO_OK if container in (".mp4", ".mov") else (
                WEBM_AUDIO_OK if container == ".webm" else None)
            if ok is None or all(c in ok for c in acodecs):
                cmd += ["-c:a", "copy"]
            elif container == ".webm":
                cmd += ["-c:a", "libopus", "-b:a", "160k"]
            else:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
        subs = info.subtitle_streams()
        if subs and container != ".webm":
            if container == ".mkv":
                cmd += ["-map", "1:s?", "-c:s", "copy"]
            else:
                text_idx = [i for i, s in enumerate(subs) if s.codec_name in TEXT_SUBS]
                for i in text_idx:
                    cmd += ["-map", f"1:s:{i}"]
                if text_idx:
                    cmd += ["-c:s", "mov_text"]
        cmd += ["-map_metadata", "1", "-map_chapters", "1"]
    if container in (".mp4", ".mov"):
        cmd += ["-movflags", "+faststart"]
    if source and info is not None and info.has_audio:
        cmd.append("-shortest")
    cmd.append(out)
    return cmd


def photo_convert_cmd(ffmpeg: str, src: str, out: str) -> List[str]:
    """Fallback decoder for odd photo formats Pillow cannot read."""
    return [ffmpeg, "-hide_banner", "-y", "-loglevel", "error", "-i", src, "-frames:v", "1", out]


def parse_progress_frame(line: str) -> Optional[int]:
    """Read 'frame=  123' from ffmpeg -stats output."""
    import re
    m = re.search(r"frame=\s*(\d+)", line)
    return int(m.group(1)) if m else None


def probe(ffprobe: str, path: str) -> MediaInfo:
    from . import engines
    res = engines.run(probe_cmd(ffprobe, path), timeout=60)
    if res.returncode != 0:
        raise ValueError(f"Could not read this video: {res.stderr.strip() or 'unknown error'}")
    return parse_probe(json.loads(res.stdout))


_ENCODER_CACHE: Optional[set] = None


def working_encoders(ffmpeg: str) -> set:
    """Encoders that are compiled in AND actually run on this PC (cached)."""
    global _ENCODER_CACHE
    if _ENCODER_CACHE is not None:
        return _ENCODER_CACHE
    from . import engines
    res = engines.run([ffmpeg, "-hide_banner", "-encoders"], timeout=30)
    listed = parse_encoders(res.stdout)
    ok = set()
    for fam in ENCODERS.values():
        for enc in fam.values():
            if enc in listed:
                if enc.startswith("lib"):
                    ok.add(enc)
                    continue
                try:
                    t = engines.run(encoder_test_cmd(ffmpeg, enc), timeout=20)
                    if t.returncode == 0:
                        ok.add(enc)
                except Exception:
                    pass
    if "libaom-av1" in listed:
        ok.add("libaom-av1")
    _ENCODER_CACHE = ok
    return ok
