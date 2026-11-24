import numpy as np
from PIL import Image

from ultraenhance import formats
from ultraenhance.estimates import check_space, human_bytes, human_time, video_temp_bytes
from ultraenhance.photo import (PhotoOptions, auto_fix, load_image, place_on_canvas, save_image,
                                sharpen, unique_output_path)
from ultraenhance.presets import compute_geometry


def test_supported_formats():
    for ext in (".jpg", ".png", ".webp", ".bmp", ".tiff", ".heic"):
        assert formats.is_photo("a" + ext.upper())
    for ext in (".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".flv"):
        assert formats.is_video("a" + ext)
    assert not formats.is_supported("a.txt")
    flt = formats.dialog_filter()
    assert "*.heic" in flt and "*.flv" in flt


def test_expand_paths(tmp_path):
    (tmp_path / "sub").mkdir()
    for n in ("a.jpg", "b.MP4", "c.txt", "sub/d.png"):
        (tmp_path / n).write_bytes(b"x")
    allf = formats.expand_paths([tmp_path])
    assert len(allf) == 3
    assert len(formats.expand_paths([tmp_path], "photo")) == 2
    assert len(formats.expand_paths([tmp_path, tmp_path / "a.jpg"])) == 3  # no duplicates


def test_unique_output_never_overwrites(tmp_path):
    src = tmp_path / "pic.jpg"
    src.write_bytes(b"x")
    p1 = unique_output_path(str(src), tmp_path, ".jpg", "4K_enhanced")
    p1.write_bytes(b"y")
    p2 = unique_output_path(str(src), tmp_path, ".jpg", "4K_enhanced")
    assert p1 != p2 and p2 != src
    # same name as source is avoided even with an empty tag
    p3 = unique_output_path(str(src), tmp_path, ".jpg", "")
    assert p3 != src


def test_place_on_canvas_modes():
    img = Image.new("RGB", (640, 480), "red")
    for mode, size in (("fit", (1440, 1080)), ("pad", (1920, 1080)), ("fill", (1920, 1080))):
        geo = compute_geometry(640, 480, "1080p Full HD", mode=mode)
        assert place_on_canvas(img, geo).size == size


def test_save_keeps_exif(tmp_path):
    src = tmp_path / "in.jpg"
    ex = Image.Exif()
    ex[0x010F] = "TestCam"
    Image.new("RGB", (32, 32), "blue").save(src, exif=ex.tobytes())
    img, exif, icc = load_image(str(src))
    for fmt in ("JPG", "PNG", "WEBP", "TIFF"):
        out = tmp_path / f"o.{fmt.lower()}"
        save_image(img, out, fmt, 90, exif, icc)
        with Image.open(out) as o:
            assert o.size == (32, 32)
            if fmt == "JPG":
                assert o.getexif().get(0x010F) == "TestCam"


def test_filters_keep_shape():
    arr = (np.random.rand(64, 48, 3) * 255).astype(np.uint8)
    assert auto_fix(arr).shape == arr.shape
    assert sharpen(arr).shape == arr.shape


def test_old_photo_enables_cleanup():
    o = PhotoOptions(model="Old photo restore").effective()
    assert o.face_restore and o.denoise and o.auto_fix and o.remove_jpeg_artifacts


def test_estimates(tmp_path):
    assert human_bytes(1536) == "1.5 KB"
    assert human_time(3725) == "1 h 2 min"
    assert video_temp_bytes(1920, 1080, 7680, 4320, 100) > video_temp_bytes(1920, 1080, 3840, 2160, 100)
    ok, _ = check_space(tmp_path, 1)
    assert ok
    ok, msg = check_space(tmp_path, 10 ** 18)
    assert not ok and "Not enough free disk space" in msg
