"""
검색 테스트 CLI.
사용법:
  python search.py "남성스럽고 야성미 넘치는 배우"   # 한 번 검색
  python search.py                                    # 대화형(여러 번 입력)
"""
import sys
import json
import numpy as np
from engine import LocalEmbedder, search

actors = json.load(open("data/actors.json", encoding="utf-8"))
embeddings = np.load("data/embeddings.npy")
embedder = LocalEmbedder()


def show(query: str, k: int = 8):
    print(f"\n🔎 검색어: \"{query}\"")
    print("-" * 56)
    results = search(query, embedder, actors, embeddings, k=k)
    for rank, (a, score) in enumerate(results, 1):
        pct = round(score * 100)
        tag = "★BEST" if rank == 1 else f" #{rank} "
        print(f"{tag}  {pct:3d}%  {a['name']} ({a['gender']}·{a['age']}세)  — {a['desc']}")


if len(sys.argv) > 1:
    show(" ".join(sys.argv[1:]))
else:
    print("검색어를 입력하세요. (그냥 엔터 치면 종료)")
    while True:
        try:
            q = input("\n검색> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(q)
