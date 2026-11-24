"""Face restoration with GFPGAN v1.4 (ONNX) + OpenCV YuNet face detection.

Runs on CPU through onnxruntime (DirectML is used automatically on Windows
when onnxruntime-directml is installed), so it works on any PC.
"""
from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from . import engines
from .config import log

# FFHQ 5-point template for a 512x512 aligned face (GFPGAN/facexlib).
TEMPLATE_512 = np.array([
    [192.98138, 239.94708],  # eye (image left)
    [318.90277, 240.19360],  # eye (image right)
    [256.63416, 314.01935],  # nose
    [201.26117, 371.41043],  # mouth corner (image left)
    [313.08905, 371.15118],  # mouth corner (image right)
], dtype=np.float32)

_session = None
_lock = threading.Lock()


class FaceRestoreUnavailable(RuntimeError):
    pass


def _get_session():
    global _session
    with _lock:
        if _session is not None:
            return _session
        model, _ = engines.face_models()
        if not model.exists():
            raise FaceRestoreUnavailable("Face restore model is missing. Please reinstall UltraEnhance Pro.")
        import onnxruntime as ort
        providers = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _session = ort.InferenceSession(str(model), sess_options=opts, providers=providers)
        log.info("GFPGAN loaded with %s", _session.get_providers())
        return _session


def detect_faces(img_bgr: np.ndarray, min_score: float = 0.75, max_side: int = 1600):
    """Return list of 5x2 landmark arrays in img coordinates."""
    _, det_path = engines.face_models()
    if not det_path.exists():
        raise FaceRestoreUnavailable("Face detection model is missing. Please reinstall UltraEnhance Pro.")
    h, w = img_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    sh, sw = small.shape[:2]
    det = cv2.FaceDetectorYN.create(str(det_path), "", (sw, sh), min_score, 0.3, 5000)
    _, faces = det.detect(small)
    out = []
    if faces is None:
        return out
    for f in faces:
        lm = f[4:14].reshape(5, 2) / s
        size = max(f[2], f[3]) / s
        if size < 24:  # too tiny to restore meaningfully
            continue
        out.append(lm.astype(np.float32))
    return out


def _restore_crop(crop_bgr: np.ndarray) -> np.ndarray:
    sess = _get_session()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None]
    inp = sess.get_inputs()[0].name
    y = sess.run(None, {inp: x.astype(np.float32)})[0][0]
    y = np.clip((y.transpose(1, 2, 0) + 1) / 2, 0, 1)
    return cv2.cvtColor((y * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def restore_faces(img_bgr: np.ndarray, strength: float = 0.8, check=None) -> tuple[np.ndarray, int]:
    """Restore every detected face. strength 0..1 blends with the original."""
    faces = detect_faces(img_bgr)
    if not faces:
        return img_bgr, 0
    result = img_bgr.astype(np.float32)
    h, w = img_bgr.shape[:2]
    mask_base = np.zeros((512, 512), np.float32)
    cv2.rectangle(mask_base, (26, 26), (486, 486), 1.0, -1)
    mask_base = cv2.GaussianBlur(mask_base, (101, 101), 0)
    for lm in faces:
        if check:
            check()
        M, _ = cv2.estimateAffinePartial2D(lm, TEMPLATE_512, method=cv2.LMEDS)
        if M is None:
            continue
        crop = cv2.warpAffine(img_bgr, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        restored = _restore_crop(crop)
        IM = cv2.invertAffineTransform(M)
        back = cv2.warpAffine(restored, IM, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpAffine(mask_base, IM, (w, h), flags=cv2.INTER_LINEAR)[..., None] * strength
        result = result * (1 - mask) + back.astype(np.float32) * mask
    return np.clip(result, 0, 255).astype(np.uint8), len(faces)


def available() -> bool:
    m, d = engines.face_models()
    return m.exists() and d.exists()
