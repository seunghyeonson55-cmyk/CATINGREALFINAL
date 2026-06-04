"""
CATING 검색 엔진 코어 (방식 B: 임베딩)
------------------------------------------------
- 배우의 '인상 설명 텍스트'를 의미 벡터(임베딩)로 바꾼다.
- 검색 문장도 같은 방식으로 벡터로 바꾼다.
- 두 벡터의 코사인 유사도로 '의미가 가까운 순' 정렬한다.

핵심: 사전에 등록되지 않은 표현("색기 있다" 등)도
      의미가 비슷하면 매칭된다. (방식 A의 한계를 넘는 지점)

임베더는 갈아끼울 수 있게 추상화했다.
- 지금: LocalEmbedder (무료, 키 없이 컴퓨터 안에서 돈다)
- 나중: OpenAIEmbedder 한 줄 추가하면 그대로 교체 가능
"""

from __future__ import annotations
import numpy as np


# ---------- 1. 임베더 (텍스트 → 벡터) ----------

class LocalEmbedder:
    """무료 로컬 한국어 임베딩 모델. API 키/비용 없음."""

    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        # 무거운 라이브러리는 실제 사용할 때만 불러온다(import 비용 절약).
        from sentence_transformers import SentenceTransformer
        print(f"[임베더] 모델 로드 중: {model_name} (처음 한 번만 다운로드)")
        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        """문장 목록 → 정규화된 벡터 행렬 (코사인 계산이 쉽도록 길이를 1로 맞춤)."""
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 길이 1로 정규화 → 내적이 곧 코사인 유사도
            show_progress_bar=False,
        )
        return vecs


class OpenAIEmbedder:
    """OpenAI 임베딩. .env의 OPENAI_API_KEY 필요. 로컬보다 정확도가 높다."""

    def __init__(self, model_name: str = "text-embedding-3-large"):
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()  # .env 파일에서 OPENAI_API_KEY 읽기
        self.client = OpenAI()  # 키는 환경변수에서 자동으로 읽음(코드에 하드코딩 X)
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.name, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # 코사인 유사도용으로 길이 1 정규화
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)


def get_embedder(provider: str = "local"):
    """provider='local'(무료) 또는 'openai'(키 필요)로 임베더를 고른다."""
    return OpenAIEmbedder() if provider == "openai" else LocalEmbedder()


# ---------- 2. 배우 → 임베딩할 텍스트 만들기 ----------

def actor_to_text(actor: dict) -> str:
    """
    배우 한 명을 '인상을 담은 한 문장'으로 만든다.
    설명(desc) + 두드러진 인상 속성을 합쳐 의미를 풍부하게 한다.
    (실제 제품에선 이 텍스트를 멀티모달 AI가 사진 보고 써준다.)
    """
    traits = actor.get("traits", {})
    top = sorted(traits.items(), key=lambda x: x[1], reverse=True)[:4]
    trait_words = ", ".join(t for t, _ in top)
    gender = "남자" if actor.get("gender") == "남" else "여자"
    return f"{actor['desc']}. {trait_words}한 인상의 {gender} 배우."


# ---------- 3. 인덱스 빌드 / 검색 ----------

def build_embeddings(actors: list[dict], embedder) -> np.ndarray:
    texts = [actor_to_text(a) for a in actors]
    return embedder.encode(texts)


def search(query: str, embedder, actors: list[dict], embeddings: np.ndarray, k: int = 10):
    """검색 문장과 가장 의미가 가까운 배우 Top-K를 (배우, 점수)로 반환."""
    qvec = embedder.encode([query])[0]          # 검색 문장 벡터 (정규화됨)
    scores = embeddings @ qvec                   # 정규화돼 있으므로 내적 = 코사인 유사도
    order = np.argsort(-scores)[:k]              # 점수 높은 순
    return [(actors[i], float(scores[i])) for i in order]


def passes_filters(actor: dict, filters: dict) -> bool:
    """배우가 정량 조건(성별/나이/키/목소리)을 모두 통과하는지."""
    if not filters:
        return True
    g = filters.get("gender")
    if g and actor.get("gender") != g:
        return False
    age = actor.get("age")
    if filters.get("age_min") is not None and (age is None or age < filters["age_min"]):
        return False
    if filters.get("age_max") is not None and (age is None or age > filters["age_max"]):
        return False
    h = actor.get("height_cm")
    if filters.get("height_min") is not None and (h is None or h < filters["height_min"]):
        return False
    if filters.get("height_max") is not None and (h is None or h > filters["height_max"]):
        return False
    voices = filters.get("voice")  # 리스트(여러 개 허용) 또는 단일 문자열
    if voices:
        if isinstance(voices, str):
            voices = [voices]
        if actor.get("voice") not in voices:
            return False
    return True


def search_filtered(query: str, embedder, actors: list[dict], embeddings: np.ndarray,
                    filters: dict | None = None, k: int = 10):
    """
    하이브리드 검색:
      1) 정량 조건(성별/나이/키/목소리)으로 후보를 먼저 거른다 (딱 떨어지는 조건).
      2) 남은 후보만 문장 의미 유사도로 순위를 매긴다 (분위기).
    검색 문장이 비어 있으면 필터 통과자를 그대로(원래 순서) 반환.
    """
    keep = [i for i, a in enumerate(actors) if passes_filters(a, filters or {})]
    if not keep:
        return []
    if not query.strip():
        return [(actors[i], None) for i in keep[:k]]
    qvec = embedder.encode([query])[0]
    sub = embeddings[keep]
    scores = sub @ qvec
    order = np.argsort(-scores)[:k]
    return [(actors[keep[j]], float(scores[j])) for j in order]
