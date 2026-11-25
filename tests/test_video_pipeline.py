"""End-to-end video pipeline with real FFmpeg and a fake AI engine (skipped without FFmpeg).

Checks: exact frame count/timing with duplicate skipping, bounded temp frames
(streamed decode, frames deleted per chunk), resume after a crash, the H.264 ->
HEVC switch above 4096 px, and cleanup/colour/sharpen options running for real.
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from ultraenhance import engines, ffmpeg_utils as fu, video
from ultraenhance.photo import EngineContext

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                reason="ffmpeg/ffprobe not installed")

FAKE_ENGINE = textwrap.dedent('''
    import os, sys, cv2, numpy as np
    a = sys.argv
    inp, out, s = a[a.index("-i") + 1], a[a.index("-o") + 1], int(a[a.index("-s") + 1])
    log = os.environ["FAKE_LOG"]
    with open(log, "a") as f:
        f.write(str(len(os.listdir(inp))) + "\\n")
    runs = sum(1 for _ in open(log))
    if os.environ.get("FAKE_FAIL_RUN") == str(runs):
        sys.exit(3)
    for n in sorted(os.listdir(inp)):
        img = cv2.imdecode(np.fromfile(os.path.join(inp, n), np.uint8), cv2.IMREAD_COLOR)
        big = cv2.resize(img, (img.shape[1] * s, img.shape[0] * s), interpolation=cv2.INTER_NEAREST)
        cv2.imencode(".png", big)[1].tofile(os.path.join(out, n))
''')


class _Job:
    def __init__(self):
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.cancel_event = threading.Event()


class Ctl:
    """Minimal JobControl stand-in that also records the most source frames ever on disk."""

    def __init__(self, in_dir_glob):
        self.job = _Job()
        self.cancelled = False
        self.max_in = 0
        self.messages = []
        self._glob = in_dir_glob

    def progress(self, p, m="", eta=None):
        self.messages.append(m)
        self.max_in = max(self.max_in, self._glob())

    def check(self):
        pass

    def attach_process(self, p):
        pass

    def detach_process(self):
        pass


def _make_video(path: Path, seconds_static=2, seconds_moving=1, size="160x96", rate=25):
    # static colour, then moving test pattern, then static again: lots of duplicates
    graph = (f"color=c=gray:s={size}:d={seconds_static}:r={rate}[a];"
             f"testsrc2=s={size}:d={seconds_moving}:r={rate}[b];"
             f"color=c=navy:s={size}:d={seconds_static}:r={rate}[c];[a][b][c]concat=n=3:v=1:a=0[v]")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex", graph, "-map", "[v]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


def _frames(path) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames,width,height,codec_name", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["streams"][0]


@pytest.fixture
def env(tmp_path, monkeypatch):
    fake = tmp_path / "fake_esrgan.py"
    fake.write_text(FAKE_ENGINE)
    log = tmp_path / "engine_runs.txt"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setattr(engines, "realesrgan_cmd",
                        lambda inp, out, model, scale, *a, **k: [sys.executable, str(fake), "-i", str(inp),
                                                                 "-o", str(out), "-s", str(scale)])
    monkeypatch.setattr(engines, "realesrgan_path", lambda: Path(sys.executable))
    monkeypatch.setattr(video, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(fu, "_ENCODER_CACHE", None)
    ctx = EngineContext(cpu=True, temp_dir=tmp_path / "temp")
    ctx.temp_dir.mkdir()
    return tmp_path, ctx, log


def _ctl(ctx):
    return Ctl(lambda: sum(1 for _ in ctx.temp_dir.glob("job_*/in/*.png")))


def test_pipeline_skips_duplicates_keeps_frames_and_bounds_temp(env):
    tmp, ctx, log = env
    src = tmp / "src.mp4"
    _make_video(src, size="320x180")   # 4x to 720p = one AI pass
    opts = video.VideoOptions(preset="720p", chunk_frames=10, output_dir=str(tmp / "out"),
                              cleanup="light", auto_color=True, sharpen=True)
    ctl = _ctl(ctx)
    out = video.enhance_video(str(src), opts, ctx, ctl)
    s = _frames(out)
    assert int(s["nb_read_frames"]) == 125 and (s["width"], s["height"]) == (1280, 720)
    upscaled = sum(int(x) for x in log.read_text().split())
    assert upscaled < 60            # 125 frames, ~25 moving + one per static chunk
    assert any("repeated frames" in m for m in ctl.messages)
    # streamed decode: never anywhere near all 125 source frames on disk at once. Tiny test
    # frames decode at ~1000 fps, so allow the throttle's polling overshoot (real frames are far slower).
    assert ctl.max_in <= (video.STREAM_AHEAD_CHUNKS + 1) * 10 + 20
    assert not list(ctx.temp_dir.glob("job_*"))   # cleaned up


def test_pipeline_resumes_after_crash_without_redoing_chunks(env, monkeypatch):
    tmp, ctx, log = env
    src = tmp / "src.mp4"
    _make_video(src, seconds_static=1, seconds_moving=1, size="320x180")
    opts = video.VideoOptions(preset="720p", chunk_frames=10, skip_duplicates=False, output_dir=str(tmp / "out"))
    monkeypatch.setenv("FAKE_FAIL_RUN", "4")
    with pytest.raises(RuntimeError):
        video.enhance_video(str(src), opts, ctx, _ctl(ctx))
    jobs = video.unfinished_jobs()
    assert len(jobs) == 1 and jobs[0]["done_chunks"] == [0, 1, 2]
    monkeypatch.delenv("FAKE_FAIL_RUN")
    resumed = video.options_from_dict(jobs[0]["options"])
    out = video.enhance_video(str(src), resumed, ctx, _ctl(ctx))
    assert int(_frames(out)["nb_read_frames"]) == 75
    runs = [int(x) for x in log.read_text().split()]
    assert sum(runs) == 30 + 10 + 45   # 3 chunks, the failed one, then chunks 4..8 only


def test_pipeline_h264_switches_to_hevc_above_4096(env, monkeypatch):
    tmp, ctx, log = env
    src = tmp / "src.mp4"
    _make_video(src, seconds_static=0.2, seconds_moving=0.2, size="480x270")
    opts = video.VideoOptions(preset="6K (3240p)", codec="H.264", quality="Fast", output_dir=str(tmp / "out"))
    plan = video.make_plan(str(src), opts, ctx)
    assert any("H.265/HEVC" in w for w in plan.warnings)
    out = video.enhance_video(str(src), opts, ctx, _ctl(ctx))
    s = _frames(out)
    assert s["codec_name"] == "hevc" and (s["width"], s["height"]) == (5760, 3240)


FAKE_RIFE = textwrap.dedent('''
    import os, shutil, sys
    a = sys.argv
    inp, out, n = a[a.index("-i") + 1], a[a.index("-o") + 1], int(a[a.index("-n") + 1])
    src = sorted(os.listdir(inp))
    for i in range(n):
        shutil.copy(os.path.join(inp, src[min(len(src) - 1, i * len(src) // n)]),
                    os.path.join(out, "%08d.png" % (i + 1)))
''')


def test_pipeline_with_frame_interpolation(env, monkeypatch):
    tmp, ctx, log = env
    rife = tmp / "fake_rife.py"
    rife.write_text(FAKE_RIFE)
    monkeypatch.setattr(engines, "rife_cmd", lambda inp, out, n, *a, **k: [sys.executable, str(rife), "-i", str(inp),
                                                                          "-o", str(out), "-n", str(n)])
    src = tmp / "src.mp4"
    _make_video(src, seconds_static=0.4, seconds_moving=0.4, size="320x180")   # 30 frames at 25 fps
    opts = video.VideoOptions(preset="720p", chunk_frames=12, interpolate=True, target_fps=50,
                              output_dir=str(tmp / "out"))
    out = video.enhance_video(str(src), opts, ctx, _ctl(ctx))
    s = _frames(out)
    assert int(s["nb_read_frames"]) == 60


def test_pipeline_face_enhance_without_models_does_not_crash(env, monkeypatch):
    tmp, ctx, log = env
    monkeypatch.setattr(video.face_restore, "available", lambda: False)
    src = tmp / "src.mp4"
    _make_video(src, seconds_static=0.2, seconds_moving=0.2, size="320x180")
    opts = video.VideoOptions(preset="720p", face_enhance=True, output_dir=str(tmp / "out"))
    out = video.enhance_video(str(src), opts, ctx, _ctl(ctx))
    assert int(_frames(out)["nb_read_frames"]) == 15   # 3 parts x 0.2 s x 25 fps
