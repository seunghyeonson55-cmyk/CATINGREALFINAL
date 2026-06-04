"""
지원서 처리 파이프라인 (STEP 2 입력단)
------------------------------------------------
파일(PDF / PNG / JPG / DOCX) 한 개를 받아서:
  1) 안에 들어있는 이미지들을 모두 꺼낸다.
  2) 그중 얼굴이 있는 이미지를 찾아 얼굴 부분만 잘라낸다 (캡쳐).
  3) 문서의 글자에서 나이·키를 자동으로 뽑는다.
     (키는 사진으로 알 수 없으므로 '지원서에 적힌 값'에서만 얻는다.)

AI(OpenAI) 호출은 여기 없다. 이 파일은 '무료·로컬' 처리만 담당.
인상 묘사 생성/임베딩은 app.py가 engine을 통해 따로 한다.
"""
from __future__ import annotations
import io
import re
import os
import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ---------- 1. 파일 → 이미지들 + 글자 ----------

def extract_images_and_text(file_bytes: bytes, filename: str) -> tuple[list[Image.Image], str]:
    """파일 한 개에서 (이미지 목록, 글자 전체)를 꺼낸다."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in IMAGE_EXTS:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return [img], ""
    if ext == ".pdf":
        return _from_pdf(file_bytes)
    if ext == ".docx":
        return _from_docx(file_bytes)
    return [], ""


def _from_pdf(file_bytes: bytes) -> tuple[list[Image.Image], str]:
    import fitz  # PyMuPDF
    images, texts = [], []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        texts.append(page.get_text())
        # (a) 페이지에 박혀있는 원본 이미지들
        for info in page.get_images(full=True):
            xref = info[0]
            try:
                base = doc.extract_image(xref)
                img = Image.open(io.BytesIO(base["image"])).convert("RGB")
                if img.width >= 80 and img.height >= 80:
                    images.append(img)
            except Exception:
                pass
        # (b) 박힌 이미지가 없으면 페이지 전체를 그림으로 렌더 (사진이 벡터/배경인 경우 대비)
        if not page.get_images(full=True):
            pix = page.get_pixmap(dpi=150)
            images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    doc.close()
    return images, "\n".join(texts)


def _from_docx(file_bytes: bytes) -> tuple[list[Image.Image], str]:
    import docx
    d = docx.Document(io.BytesIO(file_bytes))
    text = "\n".join(p.text for p in d.paragraphs)
    images = []
    for rel in d.part.rels.values():
        if "image" in rel.reltype:
            try:
                blob = rel.target_part.blob
                img = Image.open(io.BytesIO(blob)).convert("RGB")
                if img.width >= 80 and img.height >= 80:
                    images.append(img)
            except Exception:
                pass
    return images, text


# ---------- 2. 얼굴 검출 + 크롭 ----------

_CASCADE = None


def _get_cascade():
    global _CASCADE
    if _CASCADE is None:
        import cv2
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        _CASCADE = cv2.CascadeClassifier(path)
    return _CASCADE


def detect_face_crop(pil_img: Image.Image, margin: float = 0.4) -> Image.Image | None:
    """이미지에서 가장 큰 얼굴을 찾아 여백을 두고 잘라 돌려준다. 없으면 None."""
    import cv2
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # 작은 사진은 키워서 검출(작은 얼굴 놓침 방지). 좌표는 다시 원본 비율로 되돌린다.
    H0, W0 = gray.shape[:2]
    scale = 1.0
    if min(H0, W0) < 500:
        scale = 500 / min(H0, W0)
        gray = cv2.resize(gray, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_LINEAR)
    gray = cv2.equalizeHist(gray)                         # 명암 보정 → 검출 안정화
    faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.08,  # 더 촘촘히 스캔(정밀)
                                            minNeighbors=6, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # 가장 큰 얼굴
    if scale != 1.0:                                      # 키운 좌표를 원본 크기로 환원
        x, y, w, h = (int(v / scale) for v in (x, y, w, h))
    mx, my = int(w * margin), int(h * margin)
    W, H = pil_img.size
    left, top = max(0, x - mx), max(0, y - my)
    right, bot = min(W, x + w + mx), min(H, y + h + my)
    return pil_img.crop((left, top, right, bot))


def find_best_face(images: list[Image.Image]):
    """이미지 목록에서 얼굴이 검출되는 첫 이미지를 골라 (얼굴크롭, 원본)을 반환.
    하나도 못 찾으면 (None, 가장 큰 이미지)."""
    for img in images:
        crop = detect_face_crop(img)
        if crop is not None:
            return crop, img
    biggest = max(images, key=lambda im: im.width * im.height) if images else None
    return None, biggest


# ---------- 3. 글자에서 나이·키 뽑기 ----------

def parse_height(text: str) -> int | None:
    """문서 글자에서 키(cm)를 찾는다. 예: '키 178', '178cm', '신장: 180'."""
    if not text:
        return None
    for pat in (r"(?:키|신장|height)\D{0,4}(1[4-9]\d|2[01]\d)",
                r"(1[4-9]\d|2[01]\d)\s*(?:cm|센티|센치)"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def parse_age(text: str) -> int | None:
    """문서 글자에서 나이를 찾는다. 예: '나이 28', '28세', '만 31세'."""
    if not text:
        return None
    for pat in (r"(?:나이|만|age)\D{0,3}([1-7]\d)\s*(?:세|살)?",
                r"\b([1-7]\d)\s*(?:세|살)\b"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 10 <= v <= 79:
                return v
    return None


def parse_specialty(text: str) -> str | None:
    """문서 글자에서 특기를 찾는다. 예: '특기: 승마, 검도', '특기 - 수영'."""
    if not text:
        return None
    m = re.search(r"특기\s*[:\-–]?\s*([^\n]{1,40})", text)
    if m:
        val = m.group(1).strip(" .,·")
        return val or None
    return None


def to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def person_key(filename: str) -> str:
    """사람 식별자를 정한다.
    - 경로에 폴더가 있으면(ZIP 안 폴더 구조) **바로 위 폴더 이름**을 사람으로 본다.
      예) 김도윤/1.jpg → 김도윤,  사진모음/박지원/03.png → 박지원.
    - 폴더가 없으면(파일만 올린 경우) 파일 이름 앞부분에서 끝 번호를 떼어 식별자로 쓴다.
      예) 김민준_1.jpg→김민준, 김민준-2→김민준, '김민준 (3)'→김민준, 김민준4→김민준."""
    norm = filename.replace("\\", "/")
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if len(parts) >= 2:                      # 폴더가 있으면 바로 위 폴더 = 사람
        return parts[-2]
    stem = os.path.splitext(os.path.basename(filename))[0]
    key = re.sub(r"[\s_\-]*\(?\d+\)?$", "", stem).strip()
    return key or stem


# ---------- 4. ZIP 펼치기 (압축 안의 지원서들) ----------

ZIP_INNER_EXTS = IMAGE_EXTS | {".pdf", ".docx"}


def _fix_zip_name(name: str) -> str:
    """맥OS 등에서 만든 ZIP의 한글 파일명이 깨져 보이는 걸 복원한다.
    (UTF-8 이름을 cp437로 잘못 읽은 경우 → 되돌려서 정상 한글로)."""
    import unicodedata
    try:
        return unicodedata.normalize("NFC", name.encode("cp437").decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def expand_zip(file_bytes: bytes, max_files: int = 5000,
               max_each: int = 200 * 1024 * 1024) -> list[tuple[str, bytes]]:
    """ZIP 안에서 처리 가능한 파일들만 골라 (파일명, 내용바이트) 목록으로 펼친다.
    - 폴더/숨김파일/맥OS 메타(__MACOSX, ._)는 건너뛴다.
    - 너무 큰 파일·개수 초과는 무시(압축폭탄 방지)."""
    import zipfile
    out = []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if "__MACOSX" in name:
                continue
            if not (info.flag_bits & 0x800):     # UTF-8 플래그 없으면 깨진 한글 복원
                name = _fix_zip_name(name)
            base = os.path.basename(name)
            if not base or base.startswith(".") or base.startswith("._"):
                continue
            if os.path.splitext(base)[1].lower() not in ZIP_INNER_EXTS:
                continue
            if info.file_size > max_each:
                continue
            try:
                # 폴더 경로를 살려서 돌려준다(사람=폴더 구분에 필요). 예: 김도윤/1.jpg
                out.append((name, z.read(info)))
            except Exception:
                pass
            if len(out) >= max_files:
                break
    return out
