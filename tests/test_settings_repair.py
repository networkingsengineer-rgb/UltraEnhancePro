from ultraenhance.config import Settings, _normalize_threads


def test_int_threads_is_repaired():
    s = Settings.from_dict({"threads": 4})
    assert s.threads == "1:4:2"
    assert isinstance(s.threads, str)


def test_bad_types_fall_back_to_defaults():
    s = Settings.from_dict({"tile_size": "abc", "jpg_quality": "90", "keep_exif": "no",
                            "language": None, "recent_files": "x", "threads": "bad"})
    assert s.tile_size == 0
    assert s.jpg_quality == 90
    assert s.keep_exif is False
    assert s.language == "English"
    assert s.recent_files == []
    assert s.threads == "1:2:2"


def test_valid_threads_kept():
    assert _normalize_threads("2:4:2") == "2:4:2"
