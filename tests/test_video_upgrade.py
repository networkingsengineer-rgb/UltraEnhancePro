"""Tests for the video upgrade: duplicate skipping, cleanup/colour/sharpen filters,
one-click modes, big-output presets/bitrates, H.264 size fallback, streaming helpers."""
from collections import namedtuple
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from ultraenhance import ffmpeg_utils as fu
from ultraenhance import frame_dedupe as dd
from ultraenhance import video
from ultraenhance.estimates import video_output_bytes, video_temp_bytes_streamed
from ultraenhance.presets import (MAX_QUALITY, MODELS, QUALITY_PRESETS, RESOLUTION_PRESETS, VIDEO_MODE_CUSTOM,
                                  VIDEO_MODES, VIDEO_RESOLUTION_PRESETS, apply_video_mode,
                                  canonical_video_preset, compute_geometry, detect_video_mode, target_box)
from ultraenhance.video import VideoOptions, options_from_dict


# ------------------------------------------------------------------ duplicate frames
def _img(v, w=160, h=90, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.full((h, w, 3), v, np.float32) + rng.normal(0, noise, (h, w, 3))
    return np.clip(a, 0, 255).astype(np.uint8)


def test_plan_unique_identical_and_changed():
    a, b = dd.thumbnail(_img(100)), dd.thumbnail(_img(160))
    ref = dd.plan_unique([a, a.copy(), a.copy(), b, b.copy(), a])
    assert ref == [0, 0, 0, 3, 3, 5]
    assert dd.unique_indices(ref) == [0, 3, 5]
    assert dd.output_positions(ref) == [0, 0, 0, 1, 1, 2]


def test_plan_unique_compares_with_last_kept_frame_not_previous():
    # A slow fade: every step is tiny, but drift from the kept frame must force a new keep.
    sigs = [dd.thumbnail(_img(100 + 0.4 * i)) for i in range(10)]
    ref = dd.plan_unique(sigs)
    assert ref[0] == 0 and len(dd.unique_indices(ref)) >= 3
    for i, r in enumerate(ref):
        assert abs(float((sigs[i] - sigs[r]).mean())) < dd.MEAN_THRESHOLD


def test_small_moving_object_is_not_a_duplicate():
    base = _img(80, 320, 180)
    moved = base.copy()
    cv2.rectangle(moved, (150, 80), (166, 96), (255, 255, 255), -1)  # small bright object
    assert not dd.is_near_duplicate(dd.thumbnail(base), dd.thumbnail(moved))


def test_light_noise_still_counts_as_duplicate():
    a = dd.thumbnail(_img(120, 640, 360, noise=2.0, seed=1))
    b = dd.thumbnail(_img(120, 640, 360, noise=2.0, seed=2))
    assert dd.is_near_duplicate(a, b)


def test_plan_unique_none_and_max_run():
    a = dd.thumbnail(_img(50))
    assert dd.plan_unique([a, None, a]) == [0, 1, 2]  # None is unique and becomes the kept frame
    assert dd.plan_unique([a] * 5, max_run=2) == [0, 0, 0, 3, 3]
    assert dd.plan_unique([]) == []


def test_find_duplicates_reads_files_with_unicode_path(tmp_path):
    d = tmp_path / "vidéo frames ü"
    d.mkdir()
    files = []
    for i, v in enumerate([10, 10, 200, 200, 10]):
        p = d / f"{i + 1:08d}.png"
        assert dd.write_image(p, _img(v))
        files.append(p)
    calls = []
    assert dd.find_duplicates(files, check=lambda: calls.append(1)) == [0, 0, 2, 2, 4]
    assert calls


# ------------------------------------------------------------------ filters
def test_cleanup_filter_strings():
    assert fu.cleanup_filters("off") == []
    light = fu.cleanup_filters("light")
    assert light[0].startswith("deblock=") and any(f.startswith("hqdn3d") for f in light)
    strong = fu.cleanup_filters("strong")
    assert strong[0].startswith("deblock=filter=strong") and any(f.startswith("nlmeans=") for f in strong)
    with pytest.raises(ValueError):
        fu.cleanup_filters("extreme")


def test_pre_filter_chain_order_and_no_double_denoise():
    f = fu.pre_filter_chain(True, True, None, "strong", True)
    assert f[0].startswith("deblock")            # before deinterlace: needs the original block grid
    assert f[1].startswith("bwdif")
    assert sum(x.startswith("hqdn3d") for x in f) == 1   # Denoise box adds nothing on top of cleanup
    assert f[-2].startswith("normalize=") and f[-1].startswith("eq=")
    # Denoise without cleanup keeps the old behaviour
    assert fu.pre_filter_chain(False, True) == ["hqdn3d=3:2:4:3"]
    assert fu.pre_filter_chain(auto_color=True) == fu.AUTO_COLOR_FILTERS


def test_auto_color_is_mild():
    norm = fu.AUTO_COLOR_FILTERS[0]
    assert "strength=0.3" in norm and "independence=0" in norm and "smoothing=" in norm


def test_scale_filter_sharpen_before_pad():
    vf = fu.scale_filter(1920, 1080, 1440, 1080, 240, 0, sharpen=True)
    assert vf.index("unsharp=") < vf.index("pad=")
    assert vf.startswith("scale=1440:1080:flags=lanczos,unsharp=")
    assert "unsharp" not in fu.scale_filter(1920, 1080, 1920, 1080, 0, 0)


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="ffmpeg not installed")
def test_filter_strings_run_in_real_ffmpeg():
    import subprocess
    vf = ",".join(fu.pre_filter_chain(True, True, None, "strong", True) + ["format=rgb24"])
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc2=s=320x240:d=0.4:r=25", "-vf", vf, "-f", "null", "-"], capture_output=True)
    assert r.returncode == 0, r.stderr
    vf2 = fu.scale_filter(640, 360, 480, 360, 80, 0, sharpen=True)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc2=s=320x240:d=0.4:r=25", "-vf", vf2, "-f", "null", "-"], capture_output=True)
    assert r.returncode == 0, r.stderr


def test_extract_cmd_resume_and_png_compression():
    cmd = fu.extract_frames_cmd("ffmpeg", "in.mp4", "o/%08d.png", Fraction(25), ["hqdn3d"], skip_frames=240)
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.endswith("select=gte(n\\,240)") and vf.startswith("hqdn3d,")
    assert cmd[cmd.index("-start_number") + 1] == "241"
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
    assert cmd[cmd.index("-pred") + 1] == "paeth"


# ------------------------------------------------------------------ modes
def test_video_modes_reference_real_models():
    for m in VIDEO_MODES.values():
        assert m["model"] in MODELS
        assert m["cleanup"] in fu.CLEANUP_LEVELS
    assert "Fast (recommended)" in MODELS and "Best quality (slow)" in MODELS


def test_apply_and_detect_modes():
    base = {k: getattr(VideoOptions(), k) for k in VideoOptions.__dataclass_fields__}
    for name in VIDEO_MODES:
        assert detect_video_mode(apply_video_mode(base, name)) == name
    restore = apply_video_mode(base, "Restore old video")
    assert restore["cleanup"] == "strong" and restore["face_enhance"] and restore["auto_color"]
    assert restore["sharpen"] and restore["denoise"] and restore["deinterlace"] == "auto"
    best = apply_video_mode(base, "Best quality (slow)")
    assert best["model"] == "Best quality (slow)" and best["quality"] == "Best" and best["cleanup"] == "light"
    restore["sharpen"] = False
    assert detect_video_mode(restore) == VIDEO_MODE_CUSTOM
    assert apply_video_mode(base, VIDEO_MODE_CUSTOM) == base
    assert detect_video_mode(base) == "Fast HD"   # defaults = Fast HD


# ------------------------------------------------------------------ options / resume
def test_options_from_dict_tolerates_old_and_bad_saves():
    old = {"preset": "4K (2160p)", "model": "Photo (realistic)", "codec": "H.264", "removed_option": 1}
    o = options_from_dict(old)
    assert o.preset == "4K (2160p)" and o.skip_duplicates and not o.face_enhance and o.cleanup == "off"
    assert options_from_dict({"cleanup": "weird", "model": "gone"}).cleanup == "off"
    assert options_from_dict({"model": "gone"}).model in MODELS
    assert options_from_dict(None) == VideoOptions()


def test_job_key_unchanged_for_new_options_at_default():
    o = VideoOptions(preset="4K (2160p)")
    key = video._job_key_options(o)
    for k in ("skip_duplicates", "face_enhance", "cleanup", "auto_color", "sharpen", "face_strength"):
        assert k not in key
    assert "cleanup" in video._job_key_options(VideoOptions(cleanup="strong"))


# ------------------------------------------------------------------ resolutions / big outputs
def test_video_resolution_list_and_aliases():
    assert list(VIDEO_RESOLUTION_PRESETS) == ["720p", "1080p Full HD", "2K (1440p QHD)", "4K (2160p UHD)",
                                              "6K (3240p)", "8K (4320p)"]
    assert VIDEO_RESOLUTION_PRESETS["6K (3240p)"] == (5760, 3240)
    assert VIDEO_RESOLUTION_PRESETS["8K (4320p)"] == (7680, 4320)
    # photo list unchanged; old video names still accepted
    assert "6K (3240p)" not in RESOLUTION_PRESETS
    for old, new in (("720p HD", "720p"), ("2K (1440p)", "2K (1440p QHD)"), ("4K (2160p)", "4K (2160p UHD)")):
        assert canonical_video_preset(old) == new
        assert target_box(old, 1920, 1080) == target_box(new, 1920, 1080)
    assert canonical_video_preset("nonsense") == "1080p Full HD"
    g = compute_geometry(1920, 1080, "6K (3240p)", None, "fit", even_dims=True)
    assert (g.canvas_w, g.canvas_h) == (5760, 3240)


def test_h264_falls_back_above_4096():
    assert fu.effective_codec("H.264", "MP4", 3840, 2160) == "H.264"
    assert fu.effective_codec("H.264", "MP4", 2160, 3840) == "H.264"     # portrait 4K is fine
    assert fu.effective_codec("H.264", "MP4", 4096, 2160) == "H.264"
    assert fu.effective_codec("H.264", "MP4", 5760, 3240) == "H.265/HEVC"
    assert fu.effective_codec("H.264", "MKV", 4320, 7680) == "H.265/HEVC"
    assert fu.effective_codec("AV1", "MP4", 7680, 4320) == "AV1"
    assert fu.effective_codec("H.264", "WEBM", 1920, 1080) == "AV1"


def test_recommended_bitrates_grow_with_size_and_quality():
    sizes = [VIDEO_RESOLUTION_PRESETS[k] for k in VIDEO_RESOLUTION_PRESETS]
    rates = [fu.recommended_bitrate_kbps(w, h, 30, "H.264") for w, h in sizes]
    assert rates == sorted(rates) and len(set(rates)) == len(rates)
    assert rates[1] == 12000 and rates[3] == 45000
    w, h = 3840, 2160
    q = [fu.recommended_bitrate_kbps(w, h, 30, "H.265/HEVC", k) for k in QUALITY_PRESETS]
    assert q == sorted(q)
    assert fu.recommended_bitrate_kbps(w, h, 60) > fu.recommended_bitrate_kbps(w, h, 30)
    assert fu.recommended_bitrate_kbps(w, h, 30, "AV1") < fu.recommended_bitrate_kbps(w, h, 30, "H.264")
    # monotonic between bands
    seq = [fu.recommended_bitrate_kbps(16 * s // 9, s) for s in range(480, 4400, 40)]
    assert seq == sorted(seq)


def test_max_quality_preset_and_args():
    assert MAX_QUALITY in QUALITY_PRESETS
    assert QUALITY_PRESETS[MAX_QUALITY][1] < QUALITY_PRESETS["Best"][1]
    for enc in ("libx264", "libx265", "libsvtav1", "libaom-av1", "h264_nvenc", "hevc_amf", "hevc_qsv"):
        assert fu.encoder_quality_args(enc, MAX_QUALITY)
    x = fu.encoder_quality_args("libx264", MAX_QUALITY, height=4320)
    assert x[x.index("-crf") + 1] == "14"   # never relaxed for size


def test_crf_rises_slightly_for_huge_frames():
    def crf(h):
        a = fu.encoder_quality_args("libx265", "Balanced", height=h)
        return int(a[a.index("-crf") + 1])
    assert crf(1080) == 20 and crf(2160) == 21 and crf(4320) == 22
    assert fu.encoder_quality_args("libx264", "Balanced", crf=18, height=4320)[-1] == "18"


def test_segment_cmd_passes_height():
    cmd = fu.segment_encode_cmd("ffmpeg", "f/%08d.png", Fraction(30), "s.mkv", "libx265", "scale=1:1", "Balanced",
                                height=4320)
    assert cmd[cmd.index("-crf") + 1] == "22"


def test_software_encoder_and_fat_limit():
    assert fu.software_encoder("H.265/HEVC") == "libx265"
    assert fu.software_encoder("AV1", {"libaom-av1"}) == "libaom-av1"
    assert fu.fs_file_limit("FAT32") == fu.FAT_LIMIT and fu.fs_file_limit("vfat") == fu.FAT_LIMIT
    assert fu.fs_file_limit("NTFS") is None and fu.fs_file_limit("exFAT") is None


def test_partition_fstype_longest_match():
    P = namedtuple("P", "mountpoint fstype")
    parts = [P("/", "ext4"), P("/media/usb", "vfat")]
    assert video.partition_fstype("/media/usb/videos/out.mp4", parts) == "vfat"
    assert video.partition_fstype("/home/me/out.mp4", parts) == "ext4"
    assert video.partition_fstype("/media/usb2/x", parts) == "ext4"


def test_output_and_temp_estimates():
    one_hour = 3600
    hd = video_output_bytes(1920, 1080, one_hour, "H.264")
    assert 4e9 < hd < 6e9                       # ~0.8 x 12 Mbps for an hour
    assert video_output_bytes(1920, 1080, 10, bitrate_kbps=8000) == 10_000_000
    eight = video_output_bytes(7680, 4320, one_hour, "H.265/HEVC", MAX_QUALITY)
    assert eight > 50e9
    # streamed temp space does not grow with length except for the encoded segments
    t = video_temp_bytes_streamed(1920, 1080, 7680, 4320, 24, 48, 0)
    assert t < 3e9
    full = video.temp_bytes_needed(1920, 1080, 3840, 2160, 216000, 216000, 60, 1, hd, streamed=False)
    streamed = video.temp_bytes_needed(1920, 1080, 3840, 2160, 216000, 216000, 60, 1, hd, streamed=True)
    assert streamed < full / 20


# ------------------------------------------------------------------ streaming helpers
def test_chunk_bounds_and_first_unfinished():
    assert video.chunk_bounds(0, 24) == (1, 24) and video.chunk_bounds(2, 24) == (49, 72)
    assert video.first_unfinished_chunk({0, 1, 2}, lambda i: True) == 3
    assert video.first_unfinished_chunk({0, 1, 2}, lambda i: i != 1) == 1
    assert video.first_unfinished_chunk(set(), lambda i: True) == 0


def test_decoder_throttle():
    assert not video.decoder_should_wait(next_needed=1, newest_written=47, ahead=48)
    assert video.decoder_should_wait(next_needed=1, newest_written=49, ahead=48)
    assert not video.decoder_should_wait(next_needed=25, newest_written=49, ahead=48)


def test_normalize_frame_names(tmp_path):
    for n in (3, 7, 12):
        (tmp_path / f"{n:08d}.png").write_bytes(bytes([n]))
    assert video.normalize_frame_names(tmp_path) == 3
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["00000001.png", "00000002.png", "00000003.png"]
    assert (tmp_path / "00000003.png").read_bytes() == bytes([12])
    assert video.normalize_frame_names(tmp_path) == 3


def test_face_stage_missing_models_never_crashes(monkeypatch, tmp_path):
    monkeypatch.setattr(video.face_restore, "available", lambda: False)
    st = video.FaceStage()
    assert not st.enabled
    assert st.run(tmp_path, 3) == 0


def test_face_stage_turns_off_on_engine_error(monkeypatch, tmp_path):
    monkeypatch.setattr(video.face_restore, "available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("onnxruntime missing")
    monkeypatch.setattr(video.face_restore, "restore_faces", boom)
    for i in range(3):
        dd.write_image(tmp_path / f"{i + 1:08d}.png", _img(90))
    st = video.FaceStage()
    assert st.run(tmp_path, 3) == 0 and not st.enabled


def test_time_estimate_reflects_fast_model():
    from ultraenhance.estimates import video_seconds
    slow = video_seconds(3840, 2160, 1000, model="realesrgan-x4plus")
    fast = video_seconds(3840, 2160, 1000, model="realesr-animevideov3")
    assert fast < slow / 3
    assert video_seconds(3840, 2160, 1000) == slow   # old call style unchanged
