"""
로컬 모델 vs OpenAI 임베딩 — 같은 검색어로 나란히 비교.
먼저 두 인덱스가 있어야 함:
  python build_index.py            (로컬)
  python build_index.py openai     (OpenAI)
"""
import json
import numpy as np
from engine import get_embedder, search

actors = json.load(open("data/actors.json", encoding="utf-8"))
emb_local = np.load("data/embeddings_local.npy")
emb_openai = np.load("data/embeddings_openai.npy")
local = get_embedder("local")
openai = get_embedder("openai")

QUERIES = [
    "남성스럽고 야성미 넘치는 배우",
    "색기 있고 퇴폐적인 분위기",
    "날티 나는 반항적인 느낌",
    "기품 있고 우아한 귀족 같은 분위기",
]


def top(emb, embedder, q, k=3):
    return [(a["name"], a["arch"], round(s * 100))
            for a, s in search(q, embedder, actors, emb, k=k)]


for q in QUERIES:
    print("\n" + "=" * 64)
    print(f'🔎 "{q}"')
    print("=" * 64)
    L = top(emb_local, local, q)
    O = top(emb_openai, openai, q)
    print(f"{'[로컬 무료]':<32}{'[OpenAI]'}")
    for (ln, la, ls), (on, oa, os) in zip(L, O):
        print(f"{ls:3d}% {ln}({la}){'':<14}{os:3d}% {on}({oa})")
