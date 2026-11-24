from fractions import Fraction

import pytest

from ultraenhance import ffmpeg_utils as fu


def _info(audio=("aac",), subs=(), interlaced=False):
    streams = [fu.StreamInfo(0, "video", "h264")]
    for i, a in enumerate(audio):
        streams.append(fu.StreamInfo(1 + i, "audio", a))
    for i, s in enumerate(subs):
        streams.append(fu.StreamInfo(10 + i, "subtitle", s))
    return fu.MediaInfo(1280, 720, Fraction(30000, 1001), 10.0, 300, interlaced, bool(audio), streams)


def test_parse_probe_basic_and_rotation():
    data = {"streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
         "avg_frame_rate": "30000/1001", "nb_frames": "600", "duration": "20.02",
         "side_data_list": [{"rotation": -90}], "field_order": "progressive"},
        {"index": 1, "codec_type": "audio", "codec_name": "aac"},
        {"index": 2, "codec_type": "subtitle", "codec_name": "subrip"}],
        "format": {"duration": "20.02"}}
    info = fu.parse_probe(data)
    assert (info.width, info.height) == (1080, 1920)
    assert info.fps == Fraction(30000, 1001)
    assert info.frames == 600 and info.has_audio
    assert [s.codec_name for s in info.subtitle_streams()] == ["subrip"]


def test_parse_probe_estimates_frames_and_interlace():
    data = {"streams": [{"codec_type": "video", "width": 720, "height": 576, "avg_frame_rate": "25/1",
                         "field_order": "tt"}], "format": {"duration": "4"}}
    info = fu.parse_probe(data)
    assert info.frames == 100 and info.interlaced and not info.has_audio


def test_parse_probe_no_video():
    with pytest.raises(ValueError):
        fu.parse_probe({"streams": [{"codec_type": "audio"}]})


def test_parse_encoders():
    text = """Encoders:
 V....D libx264              libx264 H.264
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder
 A....D aac                  AAC
"""
    assert fu.parse_encoders(text) == {"libx264", "h264_nvenc", "aac"}


def test_choose_encoder_prefers_hardware():
    avail = {"libx264", "libx265", "hevc_nvenc", "h264_amf"}
    assert fu.choose_encoder("H.265/HEVC", avail) == "hevc_nvenc"
    assert fu.choose_encoder("H.264", avail) == "h264_amf"
    assert fu.choose_encoder("H.264", avail, "software") == "libx264"
    assert fu.choose_encoder("H.264", avail, "qsv") == "libx264"  # falls back
    assert fu.choose_encoder("AV1", {"libsvtav1"}) == "libsvtav1"
    with pytest.raises(ValueError):
        fu.choose_encoder("VP8", avail)


@pytest.mark.parametrize("enc", ["libx264", "libx265", "libsvtav1", "h264_nvenc", "hevc_amf", "av1_qsv"])
@pytest.mark.parametrize("quality", ["Fast", "Balanced", "Best"])
def test_quality_args(enc, quality):
    args = fu.encoder_quality_args(enc, quality)
    assert args and all(isinstance(a, str) for a in args)
    br = fu.encoder_quality_args(enc, quality, bitrate_kbps=8000)
    assert "8000k" in br


def test_crf_override():
    assert fu.encoder_quality_args("libx264", "Balanced", crf=18)[-1] == "18"


def test_extract_cmd():
    cmd = fu.extract_frames_cmd("ffmpeg", "in.mp4", "out/%08d.png", Fraction(30000, 1001),
                                ["bwdif"], start=5.0, duration=5.0)
    assert cmd[cmd.index("-ss") + 1] == "5.000"
    assert cmd.index("-ss") < cmd.index("-i")  # fast input seek
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("bwdif,") and "fps=30000/1001" in vf
    assert cmd[cmd.index("-fps_mode") + 1] == "cfr"
    assert cmd[-1] == "out/%08d.png"


def test_pre_filters():
    f = fu.pre_filter_chain(True, True, "C:\\tmp\\t.trf")
    assert f[0].startswith("bwdif")
    assert "vidstabtransform" in f[1] and "C\\:/tmp/t.trf" in f[1]
    assert f[2].startswith("hqdn3d")
    assert fu.pre_filter_chain() == []


def test_vidstab_detect_cmd():
    cmd = fu.vidstab_detect_cmd("ffmpeg", "a.mp4", "C:\\x\\t.trf")
    assert "vidstabdetect" in cmd[cmd.index("-vf") + 1]
    assert cmd[-1] == "-"


def test_scale_filter_modes():
    assert fu.scale_filter(1440, 1080, 1440, 1080, 0, 0) == "scale=1440:1080:flags=lanczos,setsar=1"
    pad = fu.scale_filter(1920, 1080, 1440, 1080, 240, 0)
    assert "pad=1920:1080:240:0" in pad
    crop = fu.scale_filter(1920, 1080, 1920, 1440, 0, -180)
    assert "crop=1920:1080:0:180" in crop


def test_segment_encode_cmd():
    cmd = fu.segment_encode_cmd("ffmpeg", "f/%08d.png", Fraction(60), "seg.mkv", "hevc_nvenc",
                                "scale=3840:2160:flags=lanczos,setsar=1", "Best")
    assert cmd[cmd.index("-framerate") + 1] == "60/1"
    assert cmd[cmd.index("-c:v") + 1] == "hevc_nvenc"
    assert "p7" in cmd and cmd[-1] == "seg.mkv"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"


def test_mux_mp4_copies_audio_and_converts_text_subs():
    info = _info(audio=("aac",), subs=("subrip", "hdmv_pgs_subtitle"))
    cmd = fu.mux_cmd("ffmpeg", "list.txt", "out.mp4", "src.mkv", info, ".mp4")
    s = " ".join(cmd)
    assert "-f concat" in s and "-c:v copy" in s
    assert "-map 1:a?" in s and "-c:a copy" in s
    assert "-map 1:s:0" in s and "1:s:1" not in s  # bitmap subs dropped for MP4
    assert "-c:s mov_text" in s and "+faststart" in s


def test_mux_mkv_copies_everything():
    info = _info(audio=("pcm_s16le",), subs=("hdmv_pgs_subtitle",))
    s = " ".join(fu.mux_cmd("ffmpeg", "l.txt", "o.mkv", "s.avi", info, ".mkv"))
    assert "-c:a copy" in s and "-map 1:s? -c:s copy" in s


def test_mux_mp4_reencodes_unsupported_audio():
    info = _info(audio=("pcm_s16le",))
    s = " ".join(fu.mux_cmd("ffmpeg", "l.txt", "o.mp4", "s.avi", info, ".mp4"))
    assert "-c:a aac" in s


def test_mux_webm_audio():
    info = _info(audio=("aac",))
    s = " ".join(fu.mux_cmd("ffmpeg", "l.txt", "o.webm", "s.mp4", info, ".webm"))
    assert "libopus" in s and "-c:s" not in s


def test_mux_preview_trims_source():
    cmd = fu.mux_cmd("ffmpeg", "l.txt", "o.mp4", "s.mp4", _info(), ".mp4", start=12.5, duration=5)
    i = cmd.index("s.mp4")
    assert cmd[i - 5:i - 1] == ["-ss", "12.500", "-t", "5.000"]


def test_concat_list_escapes():
    txt = fu.concat_list_text(["C:\\a b\\seg_0.mkv", "D:\\it's\\seg_1.mkv"])
    assert "file 'C:/a b/seg_0.mkv'" in txt
    assert "it'\\''s" in txt


def test_progress_parse():
    assert fu.parse_progress_frame("frame=  123 fps= 30 q=-0.0 size=N/A") == 123
    assert fu.parse_progress_frame("nothing") is None
