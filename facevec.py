"""
얼굴 '시각 지문'(face recognition embedding) — 텍스트 묘사가 아니라 얼굴 자체를 벡터로.
- OpenCV의 YuNet(얼굴검출) + SFace(얼굴인식)로 128차원 얼굴 벡터를 만든다.
- 같은/닮은 얼굴일수록 코사인 유사도가 1에 가까움 → 이미지 검색에서 동일 인물이 1등.
- 모델은 OpenCV Zoo에서 1회 내려받아 캐시(추가 pip 설치 불필요).
- 초상권: '동의하고 등록한 배우 사진'과 '감독이 권리를 가진 레퍼런스 사진'에만 쓴다.
"""
from __future__ import annotations
import io
import os
import ssl
import urllib.request
import numpy as np

_YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
              "face_detection_yunet/face_detection_yunet_2023mar.onnx")
_SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
              "face_recognition_sface/face_recognition_sface_2021dec.onnx")
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".facemodels")

_det = None
_rec = None
_load_failed = False


def _ssl_ctx():
    """certifi 인증서로 SSL 컨텍스트. 없으면 기본, 그래도 실패 시 미검증 폴백은 호출부에서."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _model_path(url: str) -> str:
    os.makedirs(_DIR, exist_ok=True)
    p = os.path.join(_DIR, os.path.basename(url))
    if os.path.exists(p) and os.path.getsize(p) > 10000:
        return p
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last = None
    for ctx in (_ssl_ctx(), ssl._create_unverified_context()):  # 인증서 → 미검증 폴백
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=90) as r:
                data = r.read()
            with open(p, "wb") as f:
                f.write(data)
            return p
        except Exception as e:
            last = e
    raise last if last else RuntimeError("model download failed")


def _load():
    """검출기·인식기를 1회 로드(모델 없으면 내려받음). 실패하면 (None, None)."""
    global _det, _rec, _load_failed
    if _load_failed:
        return None, None
    if _rec is None:
        try:
            import cv2
            yp = _model_path(_YUNET_URL)
            sp = _model_path(_SFACE_URL)
            _det = cv2.FaceDetectorYN.create(yp, "", (320, 320),
                                             score_threshold=0.6)
            _rec = cv2.FaceRecognizerSF.create(sp, "")
        except Exception:
            _load_failed = True
            return None, None
    return _det, _rec


def available() -> bool:
    return _load()[1] is not None


def face_vector(img_bytes: bytes):
    """이미지 바이트 → 가장 큰 얼굴의 시각 벡터(128-d, L2 정규화). 얼굴 없으면 None."""
    det, rec = _load()
    if rec is None or not img_bytes:
        return None
    try:
        import cv2
        from PIL import Image
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        det.setInputSize((w, h))
        _n, faces = det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: float(f[2]) * float(f[3]))  # 가장 큰 얼굴
        aligned = rec.alignCrop(bgr, face)
        feat = np.asarray(rec.feature(aligned)).flatten().astype(np.float32)
        nrm = float(np.linalg.norm(feat))
        return feat / nrm if nrm > 0 else feat
    except Exception:
        return None


def cosine(a, b) -> float:
    """두 정규화 벡터의 코사인 유사도(=내적). 같은 사람이면 보통 0.4~0.9."""
    if a is None or b is None:
        return -1.0
    try:
        return float(np.dot(np.asarray(a, dtype=np.float32),
                            np.asarray(b, dtype=np.float32)))
    except Exception:
        return -1.0
